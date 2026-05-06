Shader "Custom/StereoEyeShader"
{
    Properties
    {
        _LeftEyeTex ("Left Eye Texture", 2D) = "white" {}
        _RightEyeTex ("Right Eye Texture", 2D) = "white" {}
        _ViewAreaMin ("View Area Min", Vector) = (0.25, 0.25, 0, 0)
        _ViewAreaMax ("View Area Max", Vector) = (0.75, 0.75, 0, 0)
        _BackgroundColor ("Background Color", Color) = (0, 0, 0, 1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        Cull Front      // draw the interior

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma multi_compile _ SINGLE_PASS_INSTANCED          // needed on Quest
            #pragma multi_compile _ UNITY_STEREO_INSTANCING_ENABLED
            
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            
            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            
            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };
            
            TEXTURE2D(_LeftEyeTex);
            SAMPLER(sampler_LeftEyeTex);
            TEXTURE2D(_RightEyeTex);
            SAMPLER(sampler_RightEyeTex);
            
            CBUFFER_START(UnityPerMaterial)
                float4 _ViewAreaMin;
                float4 _ViewAreaMax;
                half4 _BackgroundColor;
            CBUFFER_END
            
            Varyings vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                
                output.positionHCS = TransformObjectToHClip(input.positionOS);
                output.uv = input.uv;
                return output;
            }
            
            half4 frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                
                // Use shader properties for viewing area
                float2 uvMin = _ViewAreaMin.xy;
                float2 uvMax = _ViewAreaMax.xy;
                
                // Check if UV is within the viewing area
                if (input.uv.x < uvMin.x || input.uv.x > uvMax.x || 
                    input.uv.y < uvMin.y || input.uv.y > uvMax.y)
                {
                    return _BackgroundColor; // Background color outside viewing area
                }
                
                // Remap UV to use full texture within the viewing area
                float2 remappedUV = (input.uv - uvMin) / (uvMax - uvMin);
                remappedUV.x = 1 - remappedUV.x; // Flip X for correct stereo rendering
                
                // Check which eye is being rendered
                if (unity_StereoEyeIndex == 0) // Left eye
                {
                    return SAMPLE_TEXTURE2D(_LeftEyeTex, sampler_LeftEyeTex, remappedUV);
                }
                else // Right eye
                {
                    return SAMPLE_TEXTURE2D(_RightEyeTex, sampler_RightEyeTex, remappedUV);
                }
            }
            ENDHLSL
        }
    }
}