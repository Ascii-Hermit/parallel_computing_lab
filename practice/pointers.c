#include <stdio.h>

int main(int argc, char *argv[]) {
    int a = 10;
    int *ptr = &a;
    printf("%d\n",a);
    printf("%d\n",&ptr);
    printf("%d\n",*ptr);
    printf("%d\n",&a);
    return 0;
}


            // name : ptr                  a
            // data : 10540 -------        10
            // addr : 10552       |------>10540