# PEFT Study


## 2024年12月24日更新

新增prompt tuning和prefix tuning两种peft方法，还是使用open-instruct框架。

目前已经测试：prefix tuning和prompt tuning可以正常运行，使用1B的模型，在54机器上测试的。

问题：这两个新的peft方法的超参不知道设置多少合适，比如`learning_rate`, `warmup_ratio `和`weight_decay`之类的。小样本测试prompt tuning，loss是上升的。

## 2024年12月25日更新

### 使用方法
1. 需要download原来的open-instruct框架
2. 把这个github仓库的文件放在指定的地方，比如`open_instruct/open_instruct/finetune_peft.py`这样，脚本放到scripts里面：`open_instruct/scripts/test_finetune_lora.sh`,还有把huggingface下载下来的数据转换为JSON格式, 这个函数需要放在`open_instruct/arrow2json.py`

3. 执行生成数据的py文件,这里面是绝对路径。数据data放到`open-instruct/data/tulu-3-sft-mixture-json`,代码在arrow2json.py中，然后把json传给`train_file`参数。

4. 执行peft脚本，需要在open_instruct路径下

```bash
bash scripts/test_finetune_lora.sh
bash scripts/test_finetune_prefix_tuning.sh
bash scripts/test_finetune_prompt_tuning.sh
```

为了少样本测试，我还随机采样了9k和90k个小样本json文件，代码在sample_90k_9k_tulu_dataset.py文件中。



### 代码逻辑修改


新加了三个参数
```py
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
- 这里有超参`num_virtual_tokens`不知道设置多少合适

```py
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



