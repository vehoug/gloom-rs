#version 430 core

in vec4 vert_color;
in vec3 vert_normal;

out vec4 frag_color;

vec3 frag_color_temp;
vec3 light_direction = normalize(vec3(0.8, -0.5, 0.6));

void main()
{
    frag_color_temp = vert_color.rgb * max(dot(vert_normal, -light_direction), 0.0);
    frag_color = vec4(frag_color_temp, vert_color.a);
}