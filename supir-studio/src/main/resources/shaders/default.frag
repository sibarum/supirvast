#version 330 core

// Default fragment shader shown in the editor at startup. Edit it and press
// Compile to rebuild the live program. The varyings below match the fixed
// vertex stage (see model.vert): change their use freely, but keep the
// declarations if you want them available.
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec2 vUv;

out vec4 fragColor;

void main() {
    // Simple lit normal-as-color shading with a checker overlay from the uv,
    // so rotation and surface parameterisation are both legible at a glance.
    vec3 n = normalize(vWorldNormal);
    vec3 lightDir = normalize(vec3(0.4, 0.8, 0.6));
    float diffuse = max(dot(n, lightDir), 0.0);
    float ambient = 0.25;

    vec3 base = 0.5 + 0.5 * n;

    vec2 cell = floor(vUv * 8.0);
    float checker = mod(cell.x + cell.y, 2.0);
    base = mix(base, base * 0.6, checker);

    fragColor = vec4(base * (ambient + diffuse), 1.0);
}
