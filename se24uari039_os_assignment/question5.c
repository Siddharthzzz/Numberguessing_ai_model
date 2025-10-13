#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

#define SIZE 10

void sort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
}

void merge(int *a, int *b, int *res, int n1, int n2) {
    int i=0, j=0, k=0;
    while (i<n1 && j<n2)
        res[k++] = (a[i] < b[j]) ? a[i++] : b[j++];
    while (i<n1) res[k++] = a[i++];
    while (j<n2) res[k++] = b[j++];
}

int main() {
    int arr[SIZE] = {9,4,7,3,1,8,6,2,5,0};
    int pipe1[2], pipe2[2];
    pipe(pipe1); pipe(pipe2);

    pid_t c1 = fork();
    if (c1 == 0) {
        close(pipe1[0]);
        sort(arr, SIZE/2);
        write(pipe1[1], arr, (SIZE/2)*sizeof(int));
        close(pipe1[1]);
        return 0;
    }

    pid_t c2 = fork();
    if (c2 == 0) {
        close(pipe2[0]);
        sort(arr+SIZE/2, SIZE/2);
        write(pipe2[1], arr+SIZE/2, (SIZE/2)*sizeof(int));
        close(pipe2[1]);
        return 0;
    }

    wait(NULL); wait(NULL);
    close(pipe1[1]); close(pipe2[1]);
    int half1[SIZE/2], half2[SIZE/2], final[SIZE];
    read(pipe1[0], half1, (SIZE/2)*sizeof(int));
    read(pipe2[0], half2, (SIZE/2)*sizeof(int));

    merge(half1, half2, final, SIZE/2, SIZE/2);
    printf("Sorted array: ");
    for (int i=0; i<SIZE; i++) printf("%d ", final[i]);
    printf("\n");
    return 0;
}
