
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    char command[100];
    while (1) {
        printf("Enter a command (or 'exit' to quit): ");
        scanf("%s", command);

        if (strcmp(command, "exit") == 0) break;

        pid_t pid = fork();
        if (pid == 0) {
            execlp(command, command, NULL);
            perror("exec failed");
            return 1;   // instead of exit(1)
        } else if (pid > 0) {
            wait(NULL);
        } else {
            perror("fork failed");
        }
    }
    return 0;
}
