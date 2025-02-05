# 写在最前

之前的[puppyapple/Chinese_LLM_From_Scratch](https://github.com/puppyapple/Chinese_LLM_From_Scratch)项目是一个学习为主的、从零开始实现中文LLM的学习记录。
里面的代码多数是`jupyter notebook`格式，为了方便代码的复用，并且梳理之前的很多学习内容，我决定整理出一个更详细的「教程版」的代码项目。
这里我依然还是选择了[Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt)作为基础框架，因为它的代码结构非常清晰，没有任何复杂的封装，无论是debug还是学习代码细节都十分方便。

项目会分**预训练、SFT、RLHF**三个阶段持续更新，主要目标是将之前的项目里的`jupyter notebook`代码转换为`python`代码，并且添加更多的注释和细节，方便自己回顾以及大家参考。
最后希望能基于`litgpt`框架构成自己的一套代码集合，方便自己后续的复用。

> 如果大家追求的是连模型代码都从零实现的那种项目，我强烈建议大家去学习[minimind](https://github.com/jingyaogong/minimind)这个项目而不是我这个。
