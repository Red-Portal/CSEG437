
#include <glm/gtc/matrix_transform.hpp> //translate, rotate, scale, lookAt, perspective, etc.
#include <glm/gtc/matrix_inverse.hpp> // inverseTranspose, etc.

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#include <CL/cl_gl.h>
#endif

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <FreeImage.h>

#ifdef __linux__
#include <GL/glx.h>
#endif

#include "parameters.h"
#include "display.h"
#include "my_OpenCL_util.h"

