#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <sys/ipc.h>
#include <sys/msg.h>

struct msg_buffer {
    long msg_type;
    char msg_text[100];
};

void caesar_encrypt(char *str) {
    for (int i = 0; str[i]; i++) {
        if (isalpha(str[i])) {
            char base = isupper(str[i]) ? 'A' : 'a';
            str[i] = (str[i] - base + 3) % 26 + base;
        }
    }
}

void caesar_decrypt(char *str) {
    for (int i = 0; str[i]; i++) {
        if (isalpha(str[i])) {
            char base = isupper(str[i]) ? 'A' : 'a';
            str[i] = (str[i] - base - 3 + 26) % 26 + base;
        }
    }
}

int main() {
    key_t key = ftok("progfile", 65);
    int msgid = msgget(key, 0666 | IPC_CREAT);

    struct msg_buffer message;

    while (1) {
        printf("Enter message (or 'over and out' to quit): ");
        fgets(message.msg_text, sizeof(message.msg_text), stdin);
        message.msg_text[strcspn(message.msg_text, "\n")] = '\0';
        message.msg_type = 1;

        caesar_encrypt(message.msg_text);
        msgsnd(msgid, &message, sizeof(message), 0);

        msgrcv(msgid, &message, sizeof(message), 1, 0);

        if (strcmp(message.msg_text, "over and out") == 0) {
            msgctl(msgid, IPC_RMID, NULL);
            printf("Queue deleted. Exiting.\n");
            break;
        }

        caesar_decrypt(message.msg_text);
        printf("Decrypted message: %s\n", message.msg_text);
    }
    return 0;
}
