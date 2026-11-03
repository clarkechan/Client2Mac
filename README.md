# ClienttoMac v4.7

> Windows 后台服务：监控检测图片目录，把新图片发送到 AI 推理服务，并按推理结果将结果归档到 OK / NG 目录。

![.NET Framework](https://img.shields.io/badge/.NET%20Framework-4.8-512BD4?logo=dotnet&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20Service-0078D6?logo=windows&logoColor=white)

## 工作流程

```
E:\检测图片\<日期>\<SN>\*.jpg
        │  FileSystemWatcher 监听新图片
        ▼
Base64 编码 → POST 到 AI 推理接口（JSON）
        │
        ▼
predicted_score > 0.5 ? NG : OK
        │
        ▼
E:\Mac检测图片\<日期>\<SN>\{OK|NG}\<图片名>_result.json
```

## 功能

- **存量补处理**：启动时扫描已有图片，跳过已生成结果的图片，只处理遗漏的
- **实时处理**：监听新图片，到达后立即推理并归档
- **OK / NG 分流**：按 `predicted_score` 阈值（0.5）写入对应目录
- **异步日志**（v4.7）：使用 `SemaphoreSlim` 串行写日志，避免多张图片并发处理时日志文件冲突；日志位于输出目录下的 `log.txt`
- **以 Windows 服务运行**：基于 `Microsoft.Extensions.Hosting` 的 `UseWindowsService()`

## 配置

目前通过 `Program.cs` 中的常量进行配置：

| 常量 | 说明 |
| --- | --- |
| `apiUrl` | AI 推理服务地址 |
| `inputPath` | 待检测图片根目录 |
| `outputPath` | 结果输出根目录 |
| `imageType` | 图片扩展名（默认 `jpg`） |

## 构建与安装

1. 用 Visual Studio 打开 `ClienttoMac 4.0.sln`，以 Release 模式生成
2. 以管理员身份注册为 Windows 服务：

```bash
sc create ClienttoMac binPath= "<生成目录>\ClienttoMac 4.0.exe" start= auto
sc start ClienttoMac
```

## License

见 [LICENSE.txt](LICENSE.txt)。
