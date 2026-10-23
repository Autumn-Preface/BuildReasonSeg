# DEMO-R1 补充测试尝试记录

以下两次是测试临时目录初始化失败，未进入 56 项断言。原始大日志仅保留本地 .tmp/demo_r1_raw_test_attempts；其哈希与摘要在此归档。最终使用已有父目录下全新 basetemp 后，57 项通过（tests_57_final.txt）。

## tests_final.txt

SHA256: 7ce3e633d2e57beff333ccfdcaaa654a2b3a5a4283df933a67a32ad48d4af5bc

```text
E           OSError: could not create numbered dir with prefix pytest- in C:\Users\ROG\AppData\Local\Packages\sandbox.{409fb0d4-f0af-4ea9-a8bf-e10490f99343}\AC\Temp\pytest-of-ROG after 10 tries
E           OSError: could not create numbered dir with prefix pytest- in C:\Users\ROG\AppData\Local\Packages\sandbox.{409fb0d4-f0af-4ea9-a8bf-e10490f99343}\AC\Temp\pytest-of-ROG after 10 tries
E           OSError: could not create numbered dir with prefix pytest- in C:\Users\ROG\AppData\Local\Packages\sandbox.{409fb0d4-f0af-4ea9-a8bf-e10490f99343}\AC\Temp\pytest-of-ROG after 10 tries
E           OSError: could not create numbered dir with prefix pytest- in C:\Users\ROG\AppData\Local\Packages\sandbox.{409fb0d4-f0af-4ea9-a8bf-e10490f99343}\AC\Temp\pytest-of-ROG after 10 tries
E           OSError: could not create numbered dir with prefix pytest- in C:\Users\ROG\AppData\Local\Packages\sandbox.{409fb0d4-f0af-4ea9-a8bf-e10490f99343}\AC\Temp\pytest-of-ROG after 10 tries
```

## tests_final_workspace.txt

SHA256: 88a04ada478cf000d5501015cb2985c122a4c5c5614159a945f29106aa2bd649

```text
E           FileNotFoundError: [WinError 3] ϵͳ�Ҳ���ָ����·����: 'C:\\D\\DeepSeekHarness\\workspace\\project\\BuildReasonSeg\\.tmp\\demo_r1_final_2329'
E           FileNotFoundError: [WinError 3] ϵͳ�Ҳ���ָ����·����: 'C:\\D\\DeepSeekHarness\\workspace\\project\\BuildReasonSeg\\.tmp\\demo_r1_final_2329'
E           FileNotFoundError: [WinError 3] ϵͳ�Ҳ���ָ����·����: 'C:\\D\\DeepSeekHarness\\workspace\\project\\BuildReasonSeg\\.tmp\\demo_r1_final_2329'
E           FileNotFoundError: [WinError 3] ϵͳ�Ҳ���ָ����·����: 'C:\\D\\DeepSeekHarness\\workspace\\project\\BuildReasonSeg\\.tmp\\demo_r1_final_2329'
E           FileNotFoundError: [WinError 3] ϵͳ�Ҳ���ָ����·����: 'C:\\D\\DeepSeekHarness\\workspace\\project\\BuildReasonSeg\\.tmp\\demo_r1_final_2329'
```
