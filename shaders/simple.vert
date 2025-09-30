#version 430 core

in layout(location = 0) vec3 position;
in layout(location = 1) vec4 color;
in layout(location = 2) vec3 normal;

out vec4 vert_color;
out vec3 vert_normal;

uniform mat4 transform;

void main()
{
    gl_Position = transform * vec4(position, 1.0f);
    vert_color = color;
    vert_normal = normal;
}