FedGVD: Efficient Federated Graph Learning via Unidirectional Distillation with Dynamic Virtual Nodes

## 项目约束

不能创建新的 `.md` 文件。

使用服务器中的虚拟环境运行。

## 连接远程服务器

| 项目 | 路径或配置 |
| --- | --- |
| 服务器 IP | `10.12.54.11` |
| SSH 端口 | `22` |
| SSH 用户名 | `hxt2024` |
| 本机 SSH 私钥路径 | `F:\ZJUT\lab server\hxt\hxt2024` |
| 私钥口令文件 | `F:\ZJUT\lab server\hxt\hxt.txt`（只用于认证，不回显内容） |
| 远程项目目录 | `/ssd-data2/hxt2024/FedGVD/` |

在本机 PowerShell 中执行：

```powershell
ssh -p 22 -i "F:\ZJUT\lab server\hxt\hxt2024" hxt2024@10.12.54.11
```

用户名是 `hxt2024`，不是 `hxt`。该私钥已加密；出现 `Enter passphrase for key` 时输入私钥口令。首次连接时核对服务器指纹，再确认连接。

登录后进入项目目录：

```bash
cd /ssd-data2/hxt2024/FedGVD/
pwd
```

`pwd` 应输出 `/ssd-data2/hxt2024/FedGVD`。连接后默认目录为 `/home/hxt2024`，需要手动切换。结束连接时执行 `exit`。

### Windows OpenSSH 私钥权限报错时

当前本机 OpenSSH 曾因私钥权限过宽报 `UNPROTECTED PRIVATE KEY FILE` / `bad permissions`，并忽略私钥。此前成功连接使用的是本机已安装的 Python `paramiko`，读取同一私钥完成认证，没有修改私钥文件或权限。

可用以下 Python 代码连接并确认远程项目目录；口令通过交互输入，代码中不保存明文口令：

```python
from getpass import getpass
import paramiko

key = paramiko.RSAKey.from_private_key_file(
    r"F:\ZJUT\lab server\hxt\hxt2024",
    password=getpass("SSH 私钥口令："),
)
client = paramiko.SSHClient()
client.load_system_host_keys()
try:
    client.connect(
        "10.12.54.11",
        port=22,
        username="hxt2024",
        pkey=key,
        allow_agent=False,
        look_for_keys=False,
        timeout=10,
    )
    _, stdout, stderr = client.exec_command(
        "cd /ssd-data2/hxt2024/FedGVD/ && pwd"
    )
    print(stdout.read().decode())
    print(stderr.read().decode())
finally:
    client.close()
```

后续对话需要连接服务器时，使用上述用户名、私钥及远程项目目录；私钥内容和口令不写入 README 或命令输出。
