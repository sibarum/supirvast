#version 330 core

// Fixed vertex shader for the studio's preview model. The editor only ever
// rewrites the fragment stage, so this stage's interface is the contract the
// fragment shader binds against: it receives the model attributes, applies the
// MVP transform, and forwards world-space position, normal, and uv as
// varyings.
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;

uniform mat4 uModel;
uniform mat4 uMvp;

out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec2 vUv;

void main() {
    vec4 world = uModel * vec4(aPosition, 1.0);
    vWorldPos = world.xyz;
    // No non-uniform scale in the studio's models, so the model matrix's
    // upper-left 3x3 is a valid normal transform.
    vWorldNormal = mat3(uModel) * aNormal;
    vUv = aUv;
    gl_Position = uMvp * vec4(aPosition, 1.0);
}
