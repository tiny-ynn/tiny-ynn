# Integrate with your application
Because tiny-ynn is header-only, integrating it with your application is extremely easy. We explain how to do it step-by-step.

## Step1/3: Include tiny_ynn.h in your application
Just add the following line:
```cpp
#include "tiny_ynn/tiny_ynn.h"
```

## Step2/3: Enable C++11 options
tiny-ynn uses C++11's core features and libraries. You must use the c++11 compliant compiler and compile with c++11-mode.

### Visual Studio(2013-)
C++11 features are enabled by default, you have nothing to do about it.

### gcc(4.8-)/clang(3.3-)
Use ```-std=c++11``` option to enable c++11-mode.

> From gcc 6.0, the default compile mode for c++ is -std=gnu++14, so you don't need to add this option.


## Step3/3: Add include path of tiny-ynn to your build system
Tell your build system where tiny-ynn exists. In gcc:

```
g++ -std=c++11 -Iyour-downloaded-path -O3 your-app.cpp -o your-app
```

> Another solution: place tiny-ynn's header files under your project root
