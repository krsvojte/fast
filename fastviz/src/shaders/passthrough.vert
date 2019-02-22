in vec3 position;
out vec3 vposition;

void main() {
    gl_Position = vec4(position,1);
    vposition = position;   
}
