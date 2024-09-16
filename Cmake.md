参考网站：https://subingwen.cn/

# Cmake的使用

## 注释

```cmake
#单行注释
#[[
多行注释
]]
```

## 基本要素

```cmake
cmake_minimum_required(VERSION 3.0)#指定的最低版本
project(project_name)#定义工程名字，还可以指定工程的版本，描述，web主页地址，支持的语言等等。
add_executable(可执行的文件名 源文件1 源文件2 ...)#可执行文件的名字自己制定，源文件名称之间可以用空格和;隔开
```



## 执行

```shell
cmake CMakeLists.txt所在的文件路径
make
```



## 个性化定制

### 定义变量

如果源文件会被反复使用，每次直接写出名字会比较麻烦，我们可以定义一个变量把文件名对应的字符串存储起来，要用到set命令

```cmake
SET(VAR [VALUE] [CACHE TYPE DOCSTRING [FORCE]])
```

-   VAR：变量名
-   VALUE：变量值

例子

```cmake
set(SRC_LIST a.cpp b.cpp c.cpp)
add_executable(a.exe ${SRC_LIST})
```



### 指定c++标准

 指定出要使用的c++标准，在c++里面的宏是DCMAKE_CXX_STANDRD

在cmake中

```cmake
set(CMAKE_CXX_STANDRD 11) #增加 -std=c++11
```

执行的时候要指出这个宏的值

```shell
cmake cmakelists.txt的路径 -DCMAKE_CXX_STANDRD=11
```



### 指定输出的路径

```cmake
set(HOME /home/tunx/test)  #定义变量
set(EXECUTABLE_OUTPUT_PATH ${HOME}/bin)#把拼接好的路径值设置给EXECUTABLE_OUTPUT_PATH这个宏，如果这个路径中的子目录不存在则会自动创建
```



## 搜索文件

如果项目里面的文件比较多，在cmakelists.txt文件里面全部罗列出来是比较麻烦的，因为我们需要搜索文件。有两种方式：

```cmake
aux_source_directory(< dir > < variable >)
```

-   dir是要搜索的目录
-   variable是把dir搜索的源文件列表存储在该variable中

例子如下：

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALC)
include_directories(${PROJECT_SOURCE_DIR}/include)
# 搜索 src 目录下的源文件
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_LIST)
add_executable(app  ${SRC_LIST})
```

第二种

```cmake
file(GLOB/GLOB_RECURSE 变量名 要搜索的文件路径和文件类型)
```

-   GLOB：将指定目录下搜索到的满足条件的所有文件生成一个列表，并将其存储到变量中
-   GLOB_RECURSE：递归搜索指定的目录，将指定目录下搜索到的满足条件的所有文件生成一个列表，并将其存储到变量中
-   /表示选择其中一个

搜索当前目录下的src目录下的源文件，存在变量中

```cmake
file(GLOB SRC $(CMAKE_CURRENT_SOURCE_DIR)/src/*.cpp)
file(GLOB HEAD $(CMAKE_CURRENT_SOURCE_DIR)/include/*.h)
```

-   CMAKE_CURRENT_SOURCE_DIR 宏表示当前访问的 CMakeLists.txt 文件所在的路径。


关于要搜索的文件路径和类型可加双引号，也可不加:

```cmake
file(GLOB SRC "$(CMAKE_CURRENT_SOURCE_DIR)/src/*.cpp")
```



## 包含头文件

在编译项目源文件的时候，很多时候都需要将源文件对应的头文件路径指定出来，这样才能保证在编译过程中编译器能够找到这些头文件，并顺利通过编译。在CMake中设置要包含的目录也很简单，通过一个命令就可以搞定了，他就是include_directories

```cmake
include_directories(head_path)
```



## 制作动态库或者静态库

### 使用命令生成库

https://subingwen.cn/linux/library

### 在cmake中，制作静态库

```cmake
add_library(库名称 STATIC 源文件1 源文件2 ...)
```

在Linux中，静态库名字分为三部分：lib+库名字+.a，此处只需要指定出库的名字就可以了，另外两部分在生成该文件的时候会自动填充。

在Windows中虽然库名和Linux格式不同，但也只需指定出名字即可。



### 制作静态库

```cmake
add_library(库名称 SHARED 源文件1 源文件2 ...
```

在Linux中，动态库名字分为三部分：lib+库名字+.so，此处只需要指定出库的名字就可以了，另外两部分在生成该文件的时候会自动填充。

在Windows中虽然库名和Linux格式不同，但也只需指定出名字即可。

### 指定库的输出

#### 动态库

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALC)
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# 设置动态库生成路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_library(calc SHARED ${SRC_LIST})
```

#### 都适用

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALC)
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# 设置动态库/静态库生成路径
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# 生成动态库
#add_library(calc SHARED ${SRC_LIST})
# 生成静态库
add_library(calc STATIC ${SRC_LIST})
```



## 链接库

### 链接静态库

```cmake
link_library(<static lib> [<static lib>...])
```

-   参数一：制定出要连接的静态库的名字
    -   可以是全名 libxxx.a
    -   也可以是xxx
-   参数2-N：其他的静态库名字

如果静态库不是系统提供的（自己做的或者第三方的）可能会出现找不到静态库的情况，此时可以将静态库的路径指定出来

```cmake
link_directories(<lib path>)
```

### 链接动态库

```cmake
target_link_libraries(
    <target> 
    <PRIVATE|PUBLIC|INTERFACE> <item>... 
    [<PRIVATE|PUBLIC|INTERFACE> <item>...]...)
```

-   target 指定要加载的库的文件的名字
    -   该文件可能是一个源文件
    -   该文件可能是一个动态库/静态库文件
    -   该文件可能是一个可执行文件
-   PRIVATE|PUBLIC|INTERFACE：动态库的访问权限，默认为PUBLIC
    -   如果各个动态库之间没有依赖关系，无需做任何设置，三者没有没有区别，一般无需指定，使用默认的 PUBLIC 即可。
    -   动态库的链接具有传递性，如果动态库 A 链接了动态库B、C，动态库D链接了动态库A，此时动态库D相当于也链接了动态库B、C，并可以使用动态库B、C中定义的方法。


```cmake
target_link_libraries(A B C)
target_link_libraries(D A)
```

-   PUBLIC：在public后面的库会被Link到前面的target中，并且里面的符号也会被导出，提供给第三方使用。
-   PRIVATE：在private后面的库仅被link到前面的target中，并且终结掉，第三方不能感知你调了啥库
-   INTERFACE：在interface后面引入的库不会被链接到前面的target中，只会导出符号。

### 两种库的不同

静态库会在生成可执行程序的链接阶段被打包到可执行程序中，所以可执行程序启动，静态库就被加载到内存中了。

动态库在生成可执行程序的链接阶段不会被打包到可执行程序中，当可执行程序被启动并且调用了动态库中的函数的时候，动态库才会被加载到内存



```cmake
add_executable(my_executable main.cpp)
target_link_libraries(my_executable PRIVATE my_dynamic_library)
```



```cmake
link_libraries(my_static_library)
add_executable(my_executable main.cpp)
```



## 日志

cmake中可以给用户显示信息 这个命令是message

```cmake
message([STATUS|WARNING|AUTHOR_WARNING|FATAL_ERROR|SEND_ERROR] "message to display" ...)
```



## 变量操作

### 追加

使用set

```cmake
set(变量名1 ${变量名1} ${变量名2} ...)
```

使用list

```cmake
list(APPEND <list> [<element> ...])

# 追加(拼接)
list(APPEND SRC_1 ${SRC_1} ${SRC_2} ${TEMP})
```



我们在通过file搜索某个目录就得到了该目录下所有的源文件，但是其中有些源文件并不是我们所需要的，比如：main.cpp 如何去除其中的main.cpp

```cmake
cmake_minimum_required(VERSION 3.0)
project(TEST)
set(TEMP "hello,world")
file(GLOB SRC_1 ${PROJECT_SOURCE_DIR}/*.cpp)
# 移除前日志
message(STATUS "message: ${SRC_1}")
# 移除 main.cpp
list(REMOVE_ITEM SRC_1 ${PROJECT_SOURCE_DIR}/main.cpp)
# 移除后日志
message(STATUS "message: ${SRC_1}")
```

list还有很多强大的功能



## 宏定义

```cmake
#include <stdio.h>
#define NUMBER  3

int main()
{
    int a = 10;
#ifdef DEBUG
    printf("我是一个程序猿, 我不会爬树...\n");
#endif
    for(int i=0; i<NUMBER; ++i)
    {
        printf("hello, GCC!!!\n");
    }
    return 0;
}
```

如果要打印日志命令如下

```shell
$ gcc test.c -DDEBUG -o app
```

cmake中我们使用

```cmake
add_definitions(-D宏名称)
```

如上面的例子，我们想定义DEBUG这个宏

```cmake
add_definition(-DDEBUG)
```



# 嵌套的cmake

```shell
$ tree
.
├── build
├── calc
│   ├── add.cpp
│   ├── CMakeLists.txt
│   ├── div.cpp
│   ├── mult.cpp
│   └── sub.cpp
├── CMakeLists.txt
├── include
│   ├── calc.h
│   └── sort.h
├── sort
│   ├── CMakeLists.txt
│   ├── insert.cpp
│   └── select.cpp
├── test1
│   ├── calc.cpp
│   └── CMakeLists.txt
└── test2
    ├── CMakeLists.txt
    └── sort.cpp

6 directories, 15 files
```

众所周知，Linux的目录是树状结构，所以嵌套的 CMake 也是一个树状结构，最顶层的 CMakeLists.txt 是根节点，其次都是子节点。因此，我们需要了解一些关于 CMakeLists.txt 文件变量作用域的一些信息：

-   根节点CMakeLists.txt中的变量全局有效
-   父节点CMakeLists.txt中的变量可以在子节点中使用
-   子节点CMakeLists.txt中的变量只能在当前节点中使用



## 添加子目录

```cmake
add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])
```

-   source_dir：指定了CMakeLists.txt源文件和代码文件的位置，其实就是指定子目录
-   binary_dir：指定了输出文件的路径，一般不需要指定，忽略即可。
-   EXCLUDE_FROM_ALL：在子路径下的目标默认不会被包含到父路径的ALL目标里，并且也会被排除在IDE工程文件之外。用户必须显式构建在子路径下的目标。



根目录的cmakelists.txt

```cmake
cmake_minimum_required(VERSION 3.0)
project(test)
# 定义变量
# 静态库生成的路径
set(LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
# 测试程序生成的路径
set(EXEC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
# 头文件目录
set(HEAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
# 静态库的名字
set(CALC_LIB calc)
set(SORT_LIB sort)
# 可执行程序的名字
set(APP_NAME_1 test1)
set(APP_NAME_2 test2)
# 添加子目录
add_subdirectory(calc)
add_subdirectory(sort)
add_subdirectory(test1)
add_subdirectory(test2)
```

在根节点对应的文件中主要做了两件事情：定义全局变量和添加子目录。

定义的全局变量主要是给子节点使用，目的是为了提高子节点中的CMakeLists.txt文件的可读性和可维护性，避免冗余并降低出差的概率。

一共添加了四个子目录，每个子目录中都有一个CMakeLists.txt文件，这样它们的父子关系就被确定下来了。



calc 目录

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALCLIB)
aux_source_directory(./ SRC)
include_directories(${HEAD_PATH})
set(LIBRARY_OUTPUT_PATH ${LIB_PATH})
add_library(${CALC_LIB} STATIC ${SRC})
```

-   第3行aux_source_directory：搜索当前目录（calc目录）下的所有源文件
-   第4行include_directories：包含头文件路径，HEAD_PATH是在根节点文件中定义的
-   第5行set：设置库的生成的路径，LIB_PATH是在根节点文件中定义的
-   第6行add_library：生成静态库，静态库名字CALC_LIB是在根节点文件中定义的



## 流程控制