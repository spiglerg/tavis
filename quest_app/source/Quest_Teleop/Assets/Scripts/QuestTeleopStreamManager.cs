// QuestTeleopStreamManager.cs - Refactored to use RenderTexture‑based stereo handling like StereoVideoController
// ===========================================================================================================
// ‑ Maintains original TCP streaming & pose reporting logic.
// ‑ Replaces CompositionLayer/TexturesExtension pathway with a stereo shader applied to a target quad.
// ‑ Uses two persistent RenderTextures (leftEyeRT, rightEyeRT) that are fed every frame via Graphics.Blit.
// ‑ Supports mono and stereo modes transparently by re‑using the same video frame when only a single camera is sent.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR;

public class QuestTeleopStreamManager : MonoBehaviour
{
    // === Inspector‑configured references ===
    [Header("XR & Scene References")]
    [SerializeField] private XROrigin xrOrigin;
    [SerializeField] private Camera xrCamera;            // The HMD camera used for FOV calculations & head pose
    [SerializeField] private GameObject targetQuad;      // Quad (or any mesh) with forward‑facing normal positioned in front of the user

    [Header("Video Settings")]
    [SerializeField] private int videoWidth  = 640;
    [SerializeField] private int videoHeight = 480;

    [Header("Stereo Shader Settings")]
    [SerializeField] private Shader stereoShader;        // Defaults to "Custom/StereoEyeShader" if left null
    [Range(10f,180f)] public float contentHorizontalFOV = 110f;
    [Range(10f,180f)] public float contentVerticalFOV   = 100f;
    public Color backgroundColor = Color.black;

    // === Networking constants ===
    private const int VIDEO_PORT  = 9500;
    private const int POSE_PORT   = 9501;
    private const int BUFFER_SIZE = 65536;               // Small buffer to minimise latency

    // === Runtime networking objects ===
    private TcpListener _videoListener;
    private TcpListener _poseListener;
    private TcpClient   _videoClient;
    private TcpClient   _poseClient;
    private NetworkStream _videoStream;
    private NetworkStream _poseStream;

    // === Render‑side resources ===
    private RenderTexture leftEyeRT;
    private RenderTexture rightEyeRT;
    private Material stereoMaterial;

    // Temporary textures reused every frame to decode JPG/PNG → Texture2D → blit
    private Texture2D _tempLeftTex;
    private Texture2D _tempRightTex;

    // Ping‑pong flag not needed – RenderTextures updated in‑place

    // === Input system actions ===
    public InputActionAsset inputActions;
    // (All actions defined in original file remain – omitted here for brevity; no behavioural change)
    private InputAction aButton, bButton, xButton, yButton;
    private InputAction leftTrigger, rightTrigger;
    private InputAction leftGrip, rightGrip;
    private InputAction leftPosAction, rightPosAction;
    private InputAction leftRotAction, rightRotAction;
    private InputAction leftThumbstickAction, rightThumbstickAction;

    // === Frame buffer for hand‑over between threads ===
    private struct FrameData
    {
        public byte[] leftFrameData;
        public int    leftFrameSize;
        public byte[] rightFrameData;
        public int    rightFrameSize;
        public bool   isStereo;
    }

    private FrameData _latestFrame;
    private readonly object _frameLock = new object();
    private bool _isConnected = false;

    // ============================================================================================
    //                                          Unity Flow                                         
    // ============================================================================================

    private IEnumerator Start()
    {
        // 1. Input setup (same as original)
        ConfigureInputActions();

        // 2. Render resources
        SetupStereoRenderTextures();
        SetContentFOVAsPercentage(1f, 1f);
        //CreateTestFrames();   // Remove or comment‑out when streaming real data

        // 3. Kick‑off connection coroutine
        yield return new WaitForSeconds(1f);
        StartCoroutine(ConnectionLoop());
    }

    private void ConfigureInputActions()
    {
        var leftMap  = inputActions.FindActionMap("XRI Left Hand");
        var rightMap = inputActions.FindActionMap("XRI Right Hand");

        xButton = leftMap.FindAction("xButton");
        yButton = leftMap.FindAction("yButton");
        leftTrigger  = leftMap.FindAction("trigger");
        leftGrip     = leftMap.FindAction("grip");
        leftPosAction  = leftMap.FindAction("pos");
        leftRotAction  = leftMap.FindAction("quat");
        leftThumbstickAction = leftMap.FindAction("thumbstick");

        aButton = rightMap.FindAction("aButton");
        bButton = rightMap.FindAction("bButton");
        rightTrigger = rightMap.FindAction("trigger");
        rightGrip    = rightMap.FindAction("grip");
        rightPosAction = rightMap.FindAction("pos");
        rightRotAction = rightMap.FindAction("quat");
        rightThumbstickAction = rightMap.FindAction("thumbstick");

        // Enable all actions
        xButton.Enable(); yButton.Enable(); leftTrigger.Enable(); leftGrip.Enable();
        leftPosAction.Enable(); leftRotAction.Enable(); leftThumbstickAction.Enable();
        aButton.Enable(); bButton.Enable(); rightTrigger.Enable(); rightGrip.Enable();
        rightPosAction.Enable(); rightRotAction.Enable(); rightThumbstickAction.Enable();
    }

    private void SetupStereoRenderTextures()
    {
        // Create RTs once
        leftEyeRT  = new RenderTexture(videoWidth, videoHeight, 0, RenderTextureFormat.ARGB32) { name = "LeftEyeRT" };
        rightEyeRT = new RenderTexture(videoWidth, videoHeight, 0, RenderTextureFormat.ARGB32) { name = "RightEyeRT" };
        leftEyeRT.Create();
        rightEyeRT.Create();

        if (stereoShader == null)
        {
            stereoShader = Shader.Find("Custom/StereoEyeShader");
        }
        if (stereoShader == null)
        {
            Debug.LogError("[QuestTeleop] StereoEyeShader not found – please include it in the project.");
            return;
        }

        stereoMaterial = new Material(stereoShader);
        stereoMaterial.SetTexture("_LeftEyeTex", leftEyeRT);
        stereoMaterial.SetTexture("_RightEyeTex", rightEyeRT);

        if (targetQuad != null)
        {
            var renderer = targetQuad.GetComponent<Renderer>();
            renderer.material = stereoMaterial;
        }
        else
        {
            Debug.LogError("[QuestTeleop] Target quad not assigned – please set in inspector.");
        }
    }

    private void CreateTestFrames()
    {
        // Create quick red/green textures to verify setup
        Texture2D leftTest  = CreateSolidColorTexture(Color.red , videoWidth, videoHeight);
        Texture2D rightTest = CreateSolidColorTexture(Color.green, videoWidth, videoHeight);
        UpdateVideoFrame(leftTest, rightTest);
        Destroy(leftTest); Destroy(rightTest);
    }

    // Restored helper so external callers (including editor scripts) can drive FOV cropping just like StereoTextureController
    public void SetContentFOVAsPercentage(float horizontalPercent, float verticalPercent)
    {
        if (xrCamera == null)
        {
            Debug.LogWarning("[QuestTeleop] xrCamera reference missing – cannot set FOV by percentage.");
            return;
        }

        // Convert camera FOV to degrees per axis
        float cameraVerticalFOV   = xrCamera.fieldOfView;
        float cameraHorizontalFOV = Camera.VerticalToHorizontalFieldOfView(cameraVerticalFOV, xrCamera.aspect);

        // Apply percentages (clamped 0‑1) then update
        contentHorizontalFOV = cameraHorizontalFOV * Mathf.Clamp01(horizontalPercent);
        contentVerticalFOV   = cameraVerticalFOV   * Mathf.Clamp01(verticalPercent);

        UpdateViewingArea();
        Debug.Log($"[QuestTeleop] Content FOV set to {contentHorizontalFOV:F1}° × {contentVerticalFOV:F1}° ({horizontalPercent:P0}, {verticalPercent:P0} of camera FOV)");
    }

    private void UpdateViewingArea()
    {
        if (stereoMaterial == null || xrCamera == null) return;

        Vector4 uvBounds = CalculateUVFromFOV(contentHorizontalFOV, contentVerticalFOV);
        stereoMaterial.SetVector("_ViewAreaMin", new Vector4(uvBounds.x, uvBounds.y, 0, 0));
        stereoMaterial.SetVector("_ViewAreaMax", new Vector4(uvBounds.z, uvBounds.w, 0, 0));
        stereoMaterial.SetColor("_BackgroundColor", backgroundColor);
    }

    private Vector4 CalculateUVFromFOV(float horizontalFOV, float verticalFOV)
    {
        float halfHFOV = horizontalFOV * 0.5f;
        float uMin = ((-halfHFOV) + 270f) / 360f;
        float uMax = ((halfHFOV)  + 270f) / 360f;
        float halfVFOV = verticalFOV * 0.5f;
        float vMin = ((-halfVFOV) + 90f) / 180f;
        float vMax = ((halfVFOV)  + 90f) / 180f;
        return new Vector4(Mathf.Clamp01(uMin), Mathf.Clamp01(vMin), Mathf.Clamp01(uMax), Mathf.Clamp01(vMax));
    }

    private void Update()
    {
        // Keep view area reactive when editing values in inspector
        UpdateViewingArea();
    }

    // ============================================================================================
    //                                      Video Frame Updates                                    
    // ============================================================================================

    private void UpdateVideoFrame(Texture leftFrame, Texture rightFrame)
    {
        // Efficient blit into RTs
        Graphics.Blit(leftFrame , leftEyeRT);
        Graphics.Blit(rightFrame, rightEyeRT);
    }

    private void UpdateVideoFrameFromBytes(byte[] leftBytes, byte[] rightBytes)
    {
        EnsureTempTextures();

        _tempLeftTex.LoadImage(leftBytes);
        _tempLeftTex.Apply();

        _tempRightTex.LoadImage(rightBytes);
        _tempRightTex.Apply();

        UpdateVideoFrame(_tempLeftTex, _tempRightTex);
    }

    private void EnsureTempTextures()
    {
        if (_tempLeftTex == null)
        {
            _tempLeftTex  = new Texture2D(videoWidth, videoHeight, TextureFormat.RGB24, false);
            _tempRightTex = new Texture2D(videoWidth, videoHeight, TextureFormat.RGB24, false);
        }
    }

    private Texture2D CreateSolidColorTexture(Color color, int width, int height)
    {
        Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        Color[] pixels = new Color[width * height];
        for (int i = 0; i < pixels.Length; ++i) pixels[i] = color;
        tex.SetPixels(pixels);
        tex.Apply();
        return tex;
    }

    // ============================================================================================
    //                                   Networking & Streaming                                    
    // ============================================================================================

    private IEnumerator ConnectionLoop()
    {
        _videoListener = new TcpListener(IPAddress.Any, VIDEO_PORT);
        _poseListener  = new TcpListener(IPAddress.Any, POSE_PORT);
        _videoListener.Start();
        _poseListener.Start();
        Debug.Log($"[QuestTeleop] Listening on {VIDEO_PORT} (video), {POSE_PORT} (pose)");

        while (true)
        {
            Cleanup();
            Debug.Log("[QuestTeleop] Waiting for client connections...");

            // Accept video client
            while (!_videoListener.Pending()) yield return null;
            _videoClient = _videoListener.AcceptTcpClient();
            ConfigureTcpClient(_videoClient);
            _videoStream = _videoClient.GetStream();

            // Accept pose client
            while (!_poseListener.Pending()) yield return null;
            _poseClient = _poseListener.AcceptTcpClient();
            ConfigureTcpClient(_poseClient);
            _poseStream = _poseClient.GetStream();

            _isConnected = true;
            Debug.Log("[QuestTeleop] Client connected on both ports");

            var videoTask = Task.Run(ReceiveVideoLoop);
            var poseTask  = Task.Run(SendPoseLoop);

            while (_isConnected && IsConnected(_videoClient) && IsConnected(_poseClient))
            {
                lock (_frameLock)
                {
                    if (_latestFrame.leftFrameData != null && _latestFrame.leftFrameSize > 0)
                    {
                        try
                        {
                            if (_latestFrame.isStereo)
                            {
                                UpdateVideoFrameFromBytes(_latestFrame.leftFrameData, _latestFrame.rightFrameData);
                            }
                            else
                            {
                                // Mono – feed same image to both eyes
                                UpdateVideoFrameFromBytes(_latestFrame.leftFrameData, _latestFrame.leftFrameData);
                            }
                        }
                        catch (Exception e)
                        {
                            Debug.LogError($"[QuestTeleop] Frame update error: {e.Message}");
                        }
                        finally
                        {
                            // Mark consumed
                            _latestFrame.leftFrameData  = null;
                            _latestFrame.rightFrameData = null;
                        }
                    }
                }
                yield return null;
            }
        }
    }

    private void ConfigureTcpClient(TcpClient client)
    {
        client.NoDelay           = true;
        client.SendBufferSize    = BUFFER_SIZE;
        client.ReceiveBufferSize = BUFFER_SIZE;
    }

    private bool IsConnected(TcpClient client)
    {
        try { return client != null && client.Connected; } catch { return false; }
    }

    private async void ReceiveVideoLoop()
    {
        byte[] headerBuffer = new byte[32];
        byte[] frameBuffer  = new byte[2 * 1024 * 1024 * 3]; // 2 MB per eye max

        try
        {
            Debug.Log("[QuestTeleop] Video receive loop started");
            while (_isConnected && _videoStream != null)
            {
                // Header: [frameId:4][timestamp:8][numCams:1][leftSize:4] + (optional) [rightSize:4]
                int baseHeaderSize = 17;
                int headerRead     = await ReadExactlyAsync(_videoStream, headerBuffer, baseHeaderSize);
                if (headerRead != baseHeaderSize) break;

                int   frameId   = BitConverter.ToInt32(headerBuffer, 0);
                long  timestamp = BitConverter.ToInt64(headerBuffer, 4);
                byte  numCams   = headerBuffer[12];
                int   leftSize  = BitConverter.ToInt32(headerBuffer, 13);
                bool  isStereo  = numCams == 2;
                int   rightSize = 0;

                if (isStereo)
                {
                    int extraRead = await ReadExactlyAsync(_videoStream, headerBuffer, 4);
                    if (extraRead != 4) break;
                    rightSize = BitConverter.ToInt32(headerBuffer, 0);
                }

                int totalExpected = leftSize + (isStereo ? rightSize : 0);
                if (totalExpected > frameBuffer.Length) { Debug.LogError("[QuestTeleop] Frame too large"); break; }

                int received = await ReadExactlyAsync(_videoStream, frameBuffer, totalExpected);
                if (received != totalExpected) break;

                // Copy to managed arrays for main thread
                lock (_frameLock)
                {
                    if (_latestFrame.leftFrameData == null || _latestFrame.leftFrameData.Length < leftSize)
                        _latestFrame.leftFrameData = new byte[leftSize];
                    Buffer.BlockCopy(frameBuffer, 0, _latestFrame.leftFrameData, 0, leftSize);
                    _latestFrame.leftFrameSize = leftSize;
                    _latestFrame.isStereo      = isStereo;

                    if (isStereo)
                    {
                        if (_latestFrame.rightFrameData == null || _latestFrame.rightFrameData.Length < rightSize)
                            _latestFrame.rightFrameData = new byte[rightSize];
                        Buffer.BlockCopy(frameBuffer, leftSize, _latestFrame.rightFrameData, 0, rightSize);
                        _latestFrame.rightFrameSize = rightSize;
                    }
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[QuestTeleop] Video receive error: {e.Message}");
        }
        _isConnected = false;
    }

    private async void SendPoseLoop()
    {
        byte[] sizeBuf = new byte[4];
        try
        {
            while (_isConnected && _poseStream != null)
            {
                // Build pose JSON (original logic retained)
                Vector3 leftPos = leftPosAction.ReadValue<Vector3>();
                Quaternion leftRot = leftRotAction.ReadValue<Quaternion>();
                Vector2 leftStick = leftThumbstickAction.ReadValue<Vector2>();

                Vector3 rightPos = rightPosAction.ReadValue<Vector3>();
                Quaternion rightRot = rightRotAction.ReadValue<Quaternion>();
                Vector2 rightStick = rightThumbstickAction.ReadValue<Vector2>();

                var pose = new
                {
                    head = new
                    {
                        pos_xyz    = new [] { xrCamera.transform.position.x, xrCamera.transform.position.y, xrCamera.transform.position.z },
                        quat_wxyz  = new [] { xrCamera.transform.rotation.w, xrCamera.transform.rotation.x, xrCamera.transform.rotation.y, xrCamera.transform.rotation.z }
                    },
                    leftController = new
                    {
                        pos_xyz    = new [] { leftPos.x,  leftPos.y,  leftPos.z },
                        quat_wxyz  = new [] { leftRot.w, leftRot.x, leftRot.y, leftRot.z },
                        X          = xButton.ReadValue<float>() > 0.5f,
                        Y          = yButton.ReadValue<float>() > 0.5f,
                        trigger    = leftTrigger.ReadValue<float>(),
                        grip       = leftGrip.ReadValue<float>(),
                        thumbstick = new [] { leftStick.x, leftStick.y }
                    },
                    rightController = new
                    {
                        pos_xyz    = new [] { rightPos.x, rightPos.y, rightPos.z },
                        quat_wxyz  = new [] { rightRot.w, rightRot.x, rightRot.y, rightRot.z },
                        A          = aButton.ReadValue<float>() > 0.5f,
                        B          = bButton.ReadValue<float>() > 0.5f,
                        trigger    = rightTrigger.ReadValue<float>(),
                        grip       = rightGrip.ReadValue<float>(),
                        thumbstick = new [] { rightStick.x, rightStick.y }
                    }
                };

                string json = JsonConvert.SerializeObject(pose);
                byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
                BitConverter.GetBytes(jsonBytes.Length).CopyTo(sizeBuf, 0);
                await _poseStream.WriteAsync(sizeBuf, 0, 4);
                await _poseStream.WriteAsync(jsonBytes, 0, jsonBytes.Length);
                await _poseStream.FlushAsync();

                await Task.Delay(16); // ~60 Hz
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[QuestTeleop] Pose send error: {e.Message}");
        }
        _isConnected = false;
    }

    // ============================================================================================
    //                                       Helper Methods                                         
    // ============================================================================================

    private async Task<int> ReadExactlyAsync(NetworkStream stream, byte[] buffer, int count)
        => await ReadExactlyAsync(stream, buffer, 0, count);

    private async Task<int> ReadExactlyAsync(NetworkStream stream, byte[] buffer, int offset, int count)
    {
        int total = 0;
        while (total < count)
        {
            int read = await stream.ReadAsync(buffer, offset + total, count - total);
            if (read == 0) break;
            total += read;
        }
        return total;
    }

    private void Cleanup()
    {
        _isConnected = false;
        try { _videoStream?.Close(); } catch { }
        try { _poseStream?.Close(); } catch { }
        try { _videoClient?.Close(); } catch { }
        try { _poseClient?.Close(); } catch { }

        _videoStream = null; _poseStream = null;
        _videoClient = null; _poseClient = null;
    }

    private void OnDestroy()
    {
        Cleanup();
        try { _videoListener?.Stop(); } catch { }
        try { _poseListener?.Stop(); } catch { }

        if (leftEyeRT  != null) leftEyeRT.Release();
        if (rightEyeRT != null) rightEyeRT.Release();
    }
}
