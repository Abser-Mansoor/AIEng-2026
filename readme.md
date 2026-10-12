# MiniGPT Notebook Quickstart

Commands and installation instructions to execute the `minigpt.ipynb` Jupyter Notebook locally or on Google Colab.

---

## 1. Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU** (recommended for training, though CPU execution is supported for testing)

---

## 2. Local Setup & Installation

### Step 1: Clone the Repository & Navigate to Directory
```bash
git clone https://github.com/your-username/minigpt.git
cd minigpt
```

### Step 2: Create and Activate a Virtual Environment

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies

```bash
pip install --upgrade pip
pip install torch numpy matplotlib jupyter notebook tqdm
```

*(Optional) If installing PyTorch with specific CUDA support, visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the correct command for your GPU driver.*

---

## 3. Running the Notebook

### Launch via Jupyter Notebook
```bash
jupyter notebook minigpt.ipynb
```

### Launch via JupyterLab
```bash
jupyter lab minigpt.ipynb
```

---

## 4. Google Colab (Alternative)

If running directly in Google Colab:

1. Upload `minigpt.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Enable GPU Acceleration: Go to **Runtime** > **Change runtime type** > Select **T4 GPU**.
3. Run the setup cell at the top of the notebook:

```python
!pip install torch numpy matplotlib tqdm
```

# Commands to run deployed GPT 2.0

## Live URL

### http://62.169.21.107/

## How to run

### use the following CURL command in CMD and change the prompt field and max tokens field to control generation.

curl -X POST http://62.169.21.107/generate -H "Content-Type: application/json" -d "{\"prompt\": \"The capital of France is \", \"max_new_tokens\": 1, \"temperature\": 0.8, \"seed\": 1}"
