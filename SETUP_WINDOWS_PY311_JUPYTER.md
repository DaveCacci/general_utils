# Windows Setup Guide (VS Code + Python 3.11.4 + Jupyter)

This guide is for non-expert users and explains how to run this repository on a Windows machine.

## Recommended approach (for non-experts)

Use **VS Code + official Python and Jupyter extensions** and let VS Code manage most steps.

- Easiest day-to-day workflow: **Option A (VS Code-managed setup)**
- Classic explicit workflow: **Option B (manual individual installation)**

Note: VS Code cannot run Python notebooks without a real Python runtime. So Python still needs to be installed, but VS Code can guide and automate interpreter/kernels better.

Note: this project uses Git LFS to manage large files.
  Git LFS Setup*:
  1. Install Git
    https://git-scm.com/
  2. Install Git LFS. In cmd terminal:
    `git lfs install`
  2. Clone the repository
    `git clone <https://github.com/username/repo_name>`
    `cd <repo_name>`
  3. Download LFS files
    `git lfs pull`
  * Common issue
    If you downloaded the ZIP, you may see files of ~1 KB.
    These are placeholders and not the actual files.
    Solution: clone the repo using Git as shown above.

---

## Option A: VS Code-managed setup (recommended)

This keeps users almost entirely inside VS Code.

### A1. Install VS Code only

1. Install VS Code from <https://code.visualstudio.com/>.
2. Open VS Code.

### A2. Install required VS Code extensions

1. Open Extensions view (`Ctrl+Shift+X`).
2. Install:
  - **Python** (Microsoft)
  - **Jupyter** (Microsoft)

### A3. Open the repository

1. **File -> Open Folder...**
2. Select this project folder (`...\NMPC`).

### A4. Let VS Code install/select Python

1. Press `Ctrl+Shift+P`.
2. Run **Python: Select Interpreter**.
3. If no interpreter is found, choose **Install Python** from the prompt/list.
4. Install/select **Python 3.11.x** (preferably 3.11.4).

### A5. Create virtual environment from VS Code command palette

1. Press `Ctrl+Shift+P`.
2. Run **Python: Create Environment**.
3. Choose:
  - **Venv**
  - Base interpreter: **Python 3.11.x**
4. Select this workspace folder as destination (creates `.venv`).

### A6. Install dependencies using VS Code terminal

Open a new terminal in VS Code (it should auto-activate `.venv`), then run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyterlab ipykernel
```

### A7. Open notebooks and let VS Code pick kernel

1. Open any `.ipynb` file.
2. In the kernel selector (top-right), choose the `.venv` interpreter.
3. Run cells.

This path minimizes interpreter detection issues because environment creation and interpreter selection are both done through VS Code commands.

---

## Option B: Manual individual installation (classic)

This is the explicit path where users install tools individually and run setup commands manually.

### B1. Install VS Code

1. Install VS Code from <https://code.visualstudio.com/>.

### B2. Install Python 3.11.4 manually

1. Download Python 3.11.4 from:
  <https://www.python.org/downloads/release/python-3114/>
2. Run installer and check **Add Python to PATH**.

### B3. Verify Python

In PowerShell:

```powershell
py -3.11 --version
```

### B4. Open repo and create virtual environment

From the repo root:

```powershell
py -3.11 -m venv .venv
```

### B5. Activate virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### B6. Install project dependencies and Jupyter tools

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyterlab ipykernel
```

### B7. Select interpreter and run notebooks in VS Code

1. Use **Python: Select Interpreter** and choose `.venv`.
2. Open `.ipynb` files and choose the `.venv` kernel.

---

## Shared Troubleshooting (Option A and B)

- `py` command not found:
  reinstall Python 3.11.4 and make sure **Add Python to PATH** is enabled.
- VS Code does not show the `.venv` interpreter:
  run **Python: Select Interpreter**, then choose `.venv\Scripts\python.exe` manually.
- Terminal does not auto-activate `.venv`:
  run `.\.venv\Scripts\Activate.ps1` in the VS Code terminal.
- Notebook kernel missing:
  install `ipykernel` in `.venv` and reopen the notebook.
- Package install errors:
  run `python -m pip install --upgrade pip` and retry installation.
