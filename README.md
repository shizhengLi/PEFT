# PEFT Study


## 2024年12月24日更新

新增prompt tuning和prefix tuning两种peft方法，还是使用open-instruct框架。

目前已经测试：prefix tuning可以正常运行，使用1B的模型，但是prompt tuning会卡在处理数据的那一步，不知道为什么。在54机器上测试的。

使用方法：

把finetune_peft.py放在和fine_tune.py一样的地方,bash脚本中指定了finetune_peft.py这个文件

### 代码逻辑修改


新加了三个参数
```
 use_prefix_tuning: bool = field(
        default=False,
        metadata={"help": "If True, will use prefix tuning to train the model."},
    )

    use_prompt_tuning: bool = field(
        default=False,
        metadata={"help": "If True, will use prompt tuning to train the model."},
    )

    num_virtual_tokens: int = field(
        default=0,
        metadata={"help": "The number of virtual tokens to use, or in other words, the prompt. For PrefixTuningConfig and PromptTuningConfig"},
    )

```

新加的代码逻辑
这里有超参`num_virtual_tokens`不知道设置多少合适

```
## Add new PEFT methods compare with the original open_instruct/finetune.py file.
    elif args.use_prefix_tuning:
        logger.info("Initializing prefix tuning model...")
        peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=args.num_virtual_tokens) # default 30
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    elif args.use_prompt_tuning:
        logger.info("Initializing prompt tuning model...")
        peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=args.num_virtual_tokens, # default 8
        prompt_tuning_init_text="Follow the instruction",
        tokenizer_name_or_path=args.tokenizer_name,
        )
```

### 脚本使用

需要把allenai/tulu-3-sft-mixture这个hf下载下来的数据集转换为json，代码在arrow2json.py中，然后把json传给`train_file`参数。

为了少样本测试，我还随机采样了9k和90k个小样本json文件，代码在sample_90k_9k_tulu_dataset.py文件中。
