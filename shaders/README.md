# Shaders

These GLSL shaders are kept as external assets so qneogeo can compare or load
video filters without embedding large shader strings in C++.

- `xbrz-freescale.glsl`: libretro xBRZ freescale shader. Exposed in the video
  filter menu as `libretro xBRZ Freescale`.
- `6xbrz.glsl`: libretro fixed 6x xBRZ shader. Exposed in the video filter
  menu as `libretro 6xBRZ`; this currently renders directly to the window.
- `frame.vert`: qneogeo's current frame rendering vertex shader.
- `frame.frag`: qneogeo's current frame rendering fragment shader, including
  nearest, linear, Super2xSaI, and xBRZ Freescale modes.

Source:

- https://github.com/libretro/glsl-shaders/tree/master/xbrz/shaders

The shader files include their original license notices.
