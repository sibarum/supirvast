#version 330 core

// Built-in GLSL fallback vertex shader, used only if the default Supir vertex
// shader can't be compiled at startup (e.g. the native toolchain is missing).
// The editable vertex stage is normally authored in Supir (see
// default.vert.supir); this mirrors it so the fallback shares the same interface.
//
// The varyings carry explicit locations so they match the (Supir-derived)
// fragment stage by location, not by name: spirv-cross names interface variables
// opaquely, and GL_ARB_separate_shader_objects lets a 3.3 core context honour
// layout(location=) on these outputs. The single mat4 uniform (uMvp) is the only
// matrix; the studio's model matrix is identity, so world position and normal are
// the raw attributes (matching the Supir default).
#extension GL_ARB_separate_shader_objects : require

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;

uniform mat4 uMvp;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vWorldNormal;
layout(location = 2) out vec2 vUv;

void main() {
    vWorldPos = aPosition;
    vWorldNormal = aNormal;
    vUv = aUv;
    gl_Position = uMvp * vec4(aPosition, 1.0);
}
