#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#define SIZE 1000000

long long arr[SIZE];
long long sum = 0;
long long diff = 0;

pthread_mutex_t sum_mutex;

void *compute_sum(void *arg) {
    long long thread_sum = 0;
    for (int i = 0; i < SIZE; i++) {
        thread_sum += arr[i];
    }
    
    pthread_mutex_lock(&sum_mutex);
    sum = thread_sum;
    pthread_mutex_unlock(&sum_mutex);
    
    return NULL;
}

void *compute_diff(void *arg) {
    long long d = arr[0];
    for (int i = 1; i < SIZE; i++) {
        d -= arr[i];
    }
    diff = d;
    return NULL;
}

int main() {
    for (int i = 0; i < SIZE; i++) {
        arr[i] = i % 10;
    }
    
    if (pthread_mutex_init(&sum_mutex, NULL) != 0) {
        return 1;
    }
    
    pthread_t t1, t2;

    pthread_create(&t1, NULL, compute_sum, NULL);
    pthread_create(&t2, NULL, compute_diff, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    pthread_mutex_destroy(&sum_mutex);

    printf("Sum = %lld\n", sum); 
    printf("Difference = %lld\n", diff);

    return 0;
}