# vC
Decoding configurations
{"max_length":400, "top_p":0.9, "top_k":50, "temperature":0.8},
                  {"max_length":400, "top_p":0.9, "top_k":50257, "temperature":1},
                  {"max_length":400, "top_p":0.95, "top_k":50257, "temperature":1},
                  {"max_length":400, "top_p":1, "top_k":50257, "temperature":0.9},
                  {"max_length":400, "top_p":1, "top_k":50257, "temperature":0.7},
                  {"max_length":400, "top_p":1, "top_k":40, "temperature":1},
                  {"max_length":400, "top_p":1, "top_k":30, "temperature":1},
                  {"max_length":400, "top_p":1, "top_k":40, "temperature":0.7}

| Version | max_length | top_p | top_k | temperature |
| ------- | ---------- | ----- | ----- | ----------- |
| v1      | 400        | 0.90  | 50    | 0.8         |
| v2      | 400        | 0.90  | 50257 | 1           |
| v3      | 400        | 0.95  | 50257 | 1           |
| v4      | 400        | 1     | 50257 | 0.9         |
| v5      | 400        | 1     | 50257 | 0.7         |
| v6      | 400        | 1     | 40    | 1           |
| v7      | 400        | 1     | 30    | 1           |
| v8      | 400        | 1     | 40    | 0.7         |

# vB v2
* When splitting sentences modify logic from `*\n\n` to `*\n`. Problem seems to appear mostly in v4 prompt.

* Removal of generated messages with `"the app"`, `the application`
* Removal of generated messages with `__`
* Addition of columns for messages: medicine keywords, statistics number, phone number, website

# vB
We sticked with 5 prompts. Removed of symbol *
| Version     | Position    | Prompt      | Symbol  |
| ----------- | ----------- | ----------- | ------- |
| v1          | start       | Messages:   | numeric |
| v2          | start       | Messages to help you quit smoking: | numeric |
| v3          | start       | Write motivational messages to encourage people to quit smoking: | numeric |
| v4          | end         | Write message like the previous ones: | numeric |
| v5          | start         | Task: Write messages that are on the same topic. | numeric |

# vA combinations
Initial combinations

| Version     | Position    | Prompt      | Symbol  |
| ----------- | ----------- | ----------- | ------- |
| v1          | start       | Messages:   | *       |
| v2          | start       | Messages:   | numbers |
| v3          | start       | Messages about how to stop smoking: | * |
| v4          | start       | Messages about how to stop smoking: | numbers |
| v5          | end         | Write similar messages: | * |
| v6          | end         | Write similar messages: | numbers |
| v7          | start       | Messages to help you quit smoking: | * |
| v8          | start       | Messages to help you quit smoking: | numbers |
| v9          | end         | Write messages like the previous ones: | * |
| v10         | end         | Write messages like the previous ones: | numbers |
