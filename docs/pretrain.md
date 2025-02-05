这篇简易的教程介绍如何从零开始训练一个小尺寸的语言模型（SLM）。
真正的「大模型」预训练其实有许多细节，可以参考知乎上大佬的这篇文章[LLM训练-pretrain - 知乎](https://zhuanlan.zhihu.com/p/718354385)。

里面的很多细节实践起来成本很高，例如数据的爬取+清洗+去重，还有配比和顺序处理等，**这部分内容这份教程里暂时不涉及，但是之后有空我会对里面可以实践的部分做一些尝试**。

这里主要使用网上开源的处理好的数据集，把单卡上预训练的流程走通。

先把代码clone到本地再开始下面的流程：
```bash
git clone https://github.com/puppyapple/MiniLM.git
```


## 数据准备
### 数据集
数据我选用的是[Chinese Fineweb Edu V2.1](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1)。
`Hugging Face`之前有一个很棒的在小语言模型上的实践（[SmolLM - blazingly fast and remarkably powerful](https://huggingface.co/blog/smollm)），他们同时开源了数据集和模型，这个`Chinese Fineweb Edu V2.1`是仿照[HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)里的数据策略仿制的**中文**版本数据子集。

![Chinese Fineweb Edu V2.1](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/smollm-corpus-2025-02-05-16-19-19.png)

由于是自己单卡跑流程，数据集太大会耗时过多，这里我选择了其中评分最高的4-5分段的子集，即使如此数据量也高达`46B`，文件大小约`70G`。

### 数据下载
这里附带说一下数据集下载方面的问题。

由于众所周知的原因，国内如果不借助🪜是无法正常访问`Hugging Face`的，`ModelScope`上的数据集又不完全和`Hugging Face`上的数据集一致（比如上面我选的这个数据集在上面就没有最新的版本），所以还是需要有一个能从`Hugging Face`下载数据集的渠道。

介绍一个目前看来还比较稳定的方案[HF-Mirror](https://hf-mirror.com/)（详细教程可以参考[如何快速下载huggingface模型——全方法总结 - 知乎](https://zhuanlan.zhihu.com/p/663712983)）。

这里我推荐里面的**方法三**，使用`hfd`的方式进行下载，参照官网里的教程设置好环境变量之后，在**本项目的根目录下**执行以下命令就可以下载数据集了：

```bash
mkdir -p data/pretrain/smol_chinese
cd data
bash hfd.sh opencsg/Fineweb-Edu-Chinese-V2.1 \
    --dataset \
    --tool aria2c \
    -x 10 \
    --include "4_5/*.parquet" \
    --local-dir ./pretrain/smol_chinese/Fineweb-Edu-Chinese-V2.1
```
> `--include`参数配合正则字符串是一个很有用的功能，可以指定下载的文件类型和范围，这里我指定下载4-5分段的文件。

我的网络环境下只用了半个多小时下载完成了这70G的数据。

这个工具支持断点续传，如果下载到一半失败了，只需要重新执行一遍命令就可以了。

### 数据预处理
这里说的数据预处理仅仅指的是将数据集预先通过`tokenizer`进行处理，生成`token_ids`序列并存成`.bin`文件，减少训练过程中数据读取的耗时。

那么我们首先要确定一个`tokenizer`。由于计划训练的模型尺寸很小（108M即0.1B），如果词表过大的话，`embedding`的参数占比会过大，所以这里我用了另一个大神开源项目[minimind](https://github.com/jingyaogong/minimind)里自己训练的小版本`tokenizer`，词表大小只有`6400`。

在项目根目录下执行以下命令下载`minimind`项目，为了方便使用把其中的`tokenizer`文件夹复制到项目根目录下：
```bash
git clone https://github.com/jingyaogong/minimind.git
cp -r minimind/model/tokenizer ./minimind_tokenizer
```

有了`tokenizer`之后，处理脚本自己写起来也简单，但要考虑**并发、乱序、分块**等因素的话还是有很多细节的。

`litgpt`有一个衍生的项目[Lightning-AI/litdata](https://github.com/Lightning-AI/litdata)，里面提供了许多数据处理的功能，本着实用至上的原则，我还是直接使用这个项目里的方法构建了一个数据预处理脚本 [prepare_pretrain_dataset.py](../prepare_pretrain_dataset.py)。

按照上面的命令下载好数据集之后，可以执行以下命令进行数据预处理：

```bash
pip install 'litgpt[all]'
pip install litdata
python prepare_pretrain_dataset.py \
    --input_dir data/pretrain/smol_chinese/ \
    --output_dir data/pretrain/smol_chinese/processed/ \
    --text_key text \
    --file_type parquet  \
    --tokenizer_path minimind_tokenizer
```
这里介绍一下我这个脚本的几个参数：
- `--input_dir`：原始数据集的目录
- `--output_dir`：处理后的数据存储目录
- `--text_key`：原数据集中用于储存文本的列名（或字段名）
- `--file_type`：原数据集的文件类型，我这里支持了最常见的`jsonl`、`csv`和`parquet`三种类型，如果需要支持其他类型，可以修改脚本自行DIY
- `--tokenizer_path`：`tokenizer`的目录
- `--fast_dev_run`：是否开启快速运行模式，如果开启，则只会处理少量数据，用于调试非常方便

⚠️其他几个需要注意的点：
1. 如我在脚本里的注释里写的：由于`litdata`的缓存机制，处理过程中会将原始数据缓存后做处理，最后会删除缓存数据，默认会使用`/tmp`目录，这里为了防止`/tmp`目录空间不足，使用`input_dir`的上一级目录作为缓存目录，大家也可以自行指定
2. 我的脚本里最后统一将`tokenizer.encode`的结果转换为`torch.int16`类型，因为我的`tokenizer`词表大小为`6400`，远远小于`int16`，所以转换为`int16`类型可以节省一半的磁盘储存。当大家使用自己的`tokenizer`时，如果词表大小大于`int16`，请自行修改脚本
3. `chunk_size`参数：这个参数我没有暴露出来，指定处理完的数据分块储存的块大小（通过tokens数量来衡量）；默认是`2049 * 8012`，即`2049 * 8012`个`tokens`，大约正好对应`int32`类型下的`64M`大小（所以最后可以看到当上面使用`int16`类型时，实际产生的数据块大小是`32M`左右）
   > 在训练真正大尺寸LLM的时候，每个数据块的大小会十分重要，这个参数的意义就会得到体现
4. `num_workers`参数：这个参数我也没有暴露出来，用于指定并发量，默认是`os.cpu_count()`，使用当前机器全部的CPU核心数

这个脚本相对比较通用，可以应对绝大多数预训练数据集的`tokenizer`预处理。

在我的`32`核机器上，处理这`70G`的数据集大概需要`100`分钟（这里我对比了下发现，`minimind`项目里自己的`tokenizer`效率其实偏低，如果使用例如`chatglm3-6b`的`tokenizer`，处理效率会大大提升）。


## 模型训练
### 模型参数
`minimind`项目作者其实也做了一些关于小模型的调研，其中提到一篇[MobileLLM](https://arxiv.org/pdf/2402.14905)的论文，分析了决定小模型性能的关键`transformer`参数，详细信息可以参考论文原文，或者`minimind`项目里的总结，这里我直接借鉴了结论来配置了模型的参数。

`litgpt`统一通过`yaml`文件来配置模型参数和训练参数，这里我构建了自己的`yaml`文件[pretrain.yaml](../configs/pretrain.yaml)，并放在了`configs`目录下。其中模型参数相关配置如下：
```yaml
model_config:
  name: MiniLM 
  hf_config: {}
  scale_embeddings: false
  block_size: 512
  padded_vocab_size: 6400 
  vocab_size: 6400
  n_layer: 16
  n_head: 16
  n_query_groups: 8
  n_embd: 768
  head_size: 48
  rotary_percentage: 1.0
  parallel_residual: false
  bias: false
  norm_class_name: RMSNorm
  mlp_class_name: LLaMAMLP
  intermediate_size: 2048
```
除了注意力头数和`embedding`以及`mlp`的`hidden_size(intermediate_size)`，模型结构其实和`llama3.1`完全一致，最后得到的模型尺寸是`108M`，即`0.1B`。

这里的参数我就不一一详细解释了，如果这些有不清楚的，建议还是先去补一下相关知识。例如结合`litgpt`源码的[litgpt/litgpt/model.py](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py)文件里的模型实现来理解各个参数含义。

### 训练参数
训练参数还是参见[pretrain.yaml](../configs/pretrain.yaml)，里面每个参数其实都有详细的注释，这里我说明几个需要注意的点：

- `global_batch_size`：全局批量大小，即所有GPU的批量大小之和，这里我设置为`640`
- `micro_batch_size`：微批量大小，即每个GPU的批量大小，这里我设置为`80`
    > 当实际只有单卡的时候，上面的设置的效果就是每累计处理了`640/80=8`个micro_batch_size时就更新一次梯度
    这两个参数需要大家自己多跑几遍执行来调整到最佳大小，以充分利用自己的GPU资源, 上面的这个设置结合我的模型尺寸，显存占用大约在22G+，接近占满一个24G的3090Ti。

- `lr_warmup_steps`：学习率预热步数，即在训练开始时，学习率从0逐渐增加到最大学习率的时间步数，这里我设置为`2000`
- `max_steps/epochs`：这两个参数目前在预训练的时候目前是无效的（后续更新会支持），只能通过`max_tokens`来控制训练长度，其实我觉得这样约束对于LLM预训练来说反而更加直观
- `max_tokens`：最大训练tokens数，即训练过程中最多训练的tokens数，这里我设置为`45000000000`，大约是上面的数据跑一个epoch的量（45B）
- `lr/min_lr`：学习率/最小学习率，这里我分别设置为`0.001`和`0.00001`。由于模型很小，前期收敛很快，大家可以自行多测试几个不同的学习率来获得收敛最快的学习率

### 启动训练
有了配置文件，使用`litgpt`训练就非常方便了：
```bash
litgpt pretrain --config configs/pretrain.yaml
```
在我的`3090Ti`上，跑完这一轮的`45B tokens`数据需要六天多（基本跑了整个春节假期🤣）：

![wandb_time](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/wandb_1-2025-02-05-16-16-58.png)

`loss`最后收敛到了2以下：
![wandb_loss](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/wandb_2-2025-02-05-16-17-32.png)

可以简单测试一下「接龙」效果：
```bash
litgpt generate ./out/pretrain/minilm/108m/final/ \
    --prompt "从前有座山，山里有座庙，庙里" \
    --max_new_tokens 300
```
![generation_result](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/generation_test-2025-02-05-16-17-55.png)

可以看到模型基本可以说人话，但是由于用的语料还是偏小，并且我只使用了4-5分段的语料，很可能导致有偏，因此很多知识都是胡编乱造的，语法问题也很多，同时也有复读机现象。

（在我之前的`TinyStories`数据集的尝试中，效果就好很多，因为语料多样性很低，基本都是小故事，所以模型学起来要容易很多）

`minimind`项目的[issues#101](https://github.com/jingyaogong/minimind/issues/101)里其实大家也有相关的讨论，这些现象经过`SFT`之后会有很大改善，分析原因可能是因为数据质量还是不够好，以及模型尺寸太小，很多能力其实是在`SFT`阶段才真正学习到的。

但无论如何，流程至此算是完全走通了，结果上看过程没有出现大的问题。

## 总结
可以看到，其实跑通预训练的流程并不难，难的是数据准备和处理，它们直接决定了训练的稳定性，以及最终的模型效果。

所以关于这部分我后续将花更多时间来研究，争取还是在小尺寸模型上做出一些有意思的尝试。

关于预训练的详细内容就这么多了，下次更新见！
