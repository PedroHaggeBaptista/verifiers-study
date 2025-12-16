# Word Count Environment para Verifiers

Um ambiente de exemplo completo para o framework [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) que demonstra como criar, testar e avaliar ambientes customizados para LLM Reinforcement Learning.

## 🎯 Sobre o Projeto

Este projeto implementa um ambiente completo end-to-end que testa a capacidade de LLMs de contar palavras em textos. Serve como:
- **Template** para criar novos ambientes no Verifiers
- **Exemplo didático** de integração completa com o framework
- **Demonstração** de sistema de recompensas multi-critério
- **Guia prático** para uso com diferentes APIs (OpenAI, Anthropic, AWS Bedrock)
- **Pipeline completo**: desde avaliação até geração de rollouts para treinamento RL

## 🔁 Fluxo Completo

```
1️⃣ CRIAR AMBIENTE          2️⃣ AVALIAR MODELO         3️⃣ GERAR ROLLOUTS       4️⃣ TREINAR COM RL
   (Verifiers)              (vf-eval + LiteLLM)      (generate_rollouts)     (Prime-RL)
       │                            │                        │                      │
       ├─> word_count.py            ├─> Claude Sonnet       ├─> rollouts.jsonl    ├─> Modelo treinado
       ├─> Rubric (rewards)         ├─> Calcula rewards     └─> Formato Prime     └─> Deploy
       └─> Parser XML               └─> Estatísticas
```

Este projeto demonstra todo o pipeline, desde a criação do ambiente até a geração de dados prontos para treinamento RL.

## 📁 Estrutura do Projeto

```
word_count_environ/
├── environments/
│   └── word_count/
│       ├── word_count.py          # 🎯 Implementação principal do ambiente
│       ├── test_environment.py    # ✅ Testes unitários
│       ├── pyproject.toml         # 📦 Configuração do pacote
│       └── README.md              # 📖 Documentação detalhada
├── rollout_generation/
│   ├── generate_rollouts.py       # 🔄 Script para gerar rollouts via LiteLLM
│   ├── README.md                  # 📖 Documentação de geração de rollouts
│   └── rollouts/
│       └── rollouts.jsonl         # 💾 Rollouts gerados (formato Prime-RL)
├── litellm/
│   └── litellm_config.yaml        # ☁️  Configuração do proxy LiteLLM
└── README.md                      # 📋 Este arquivo
```

## 🚀 Quick Start: 4 Passos para RL

### Passo 1: Instalar Ambiente

```bash
cd environments/word_count
uv pip install -e .
```

### Passo 2: Configurar LiteLLM (para AWS Bedrock)

```bash
# Configurar credenciais AWS
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret  
export AWS_SESSION_TOKEN=your_token  # se aplicável

# Iniciar proxy (em outro terminal)
litellm --config litellm/litellm_config.yaml --port 8000
```

### Passo 3: Avaliar Modelo

```bash
cd ../..  # voltar para raiz
uv run vf-eval word-count -m claude-sonnet -b http://localhost:8000 -n 5 -r 1
```

### Passo 4: Gerar Rollouts para Treinamento

```bash
cd rollout_generation
uv run generate_rollouts.py
# ✅ Rollouts salvos em: rollouts/rollouts.jsonl
```

**Pronto!** Você tem dados prontos para treinar com [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl) 🎉

---

## 🧪 Testes e Validação

### Testes Unitários

Valide que o ambiente está funcionando corretamente:

```bash
cd environments/word_count
uv run test_environment.py
```

**Saída esperada:**
```
> uv run test_environment.py
Running tests...

Map: 100%|████████████████████████████████████████████████████| 5/5 [00:00<00:00, 496.76 examples/s]
✅ Environment created successfully!
Map: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00, 1543.44 examples/s]
✅ Dataset structure is correct!
Map: 100%|████████████████████████████████████████████████████| 1/1 [00:00<00:00, 780.92 examples/s]
✅ Parser works correctly!

🎉 All tests passed!
```

## 📋 Características do Ambiente

### Dataset
- **Geração sintética** de textos com contagem controlada de palavras
- **Configurável**: número de exemplos, min/max palavras por texto
- **Formato Verifiers**: campos `question`, `answer`, `info`, `task`

### Parser
- **XMLParser** com tags `<word_count>N</word_count>`
- Extrai respostas numéricas de forma estruturada
- Tolerante a variações de formatação

### Sistema de Recompensas (Rubric)

| Função | Peso | Descrição |
|--------|------|-----------|
| **Exact Match** | 1.0 | Resposta exata (1.0 ou 0.0) |
| **Format** | 0.2 | Uso correto das tags XML |
| **Partial Credit** | 0.1 | Crédito parcial: 1.0 (exato), 0.5 (±1), 0.2 (±2) |

**Recompensa total máxima:** 1.3 (1.0 + 0.2 + 0.1)

## 🌐 Avaliação com LLMs via API

O framework Verifiers usa APIs compatíveis com OpenAI para fazer chamadas aos modelos. Para APIs que não seguem esse padrão (como AWS Bedrock), é necessário usar um **proxy de tradução**.

### Por que LiteLLM?

**LiteLLM** é um proxy universal que converte requisições OpenAI para diversos provedores:
- ✅ AWS Bedrock (Amazon)
- ✅ Vertex AI (Google)
- ✅ Azure OpenAI
- ✅ Anthropic, Cohere, e 100+ outros

Isso permite usar o `vf-eval` com qualquer provedor sem modificar código.

### Configuração do LiteLLM

#### 1. Instalar LiteLLM

```bash
pip install 'litellm[proxy]'
```

#### 2. Criar arquivo de configuração

Crie `litellm/litellm_config.yaml`:

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
      aws_region_name: us-east-1
```

#### 3. Configurar credenciais AWS

```bash
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key
export AWS_SESSION_TOKEN=your_session_token  # se aplicável
```

#### 4. Iniciar o proxy

```bash
litellm --config litellm/litellm_config.yaml --port 8000
```

O proxy estará rodando em `http://localhost:8000` e traduzirá chamadas OpenAI para AWS Bedrock.

### Executando a Avaliação

Com o proxy LiteLLM rodando, execute:

```bash
uv run vf-eval word-count \
  -m claude-sonnet \
  -b http://localhost:8000 \
  -k LITELLM_API_KEY \
  -n 5 -r 1
```

**Parâmetros:**
- `-m`: Nome do modelo (definido no `litellm_config.yaml`)
- `-b`: URL base do proxy LiteLLM
- `-k`: Variável de ambiente com a chave API (pode ser dummy para Bedrock)
- `-n`: Número de exemplos a avaliar
- `-r`: Rollouts por exemplo (quantas vezes avaliar cada exemplo)

### Exemplo de Resultado Real

```
2025-12-16 00:35:19 - verifiers.utils.env_utils - INFO - Loading environment: word-count
2025-12-16 00:35:19 - verifiers.utils.env_utils - INFO - Using default args: min_words=5, seed=42, max_words=50, num_examples=100
Map: 100%|████████████████████████████████| 100/100 [00:00<00:00, 7713.52 examples/s]
2025-12-16 00:35:19 - verifiers.utils.env_utils - INFO - Successfully loaded environment 'word-count'
2025-12-16 00:35:19 - verifiers.utils.eval_utils - INFO - Starting evaluation with model: claude-sonnet
2025-12-16 00:35:19 - verifiers.utils.eval_utils - INFO - Configuration: num_examples=5, rollouts_per_example=1, max_concurrent=32
Processing 5 groups (5 total rollouts): 100%|████████████| 5/5 [00:02<00:00,  2.12it/s]
2025-12-16 00:35:21 - verifiers.utils.eval_utils - INFO - Evaluation completed in 2.36 seconds

--- Evaluation ---
Environment: word-count
Model: claude-sonnet
Provider: http://localhost:8000
Examples: 5
Rollouts per example: 1

--- Example ---
╭───────────────────────────────────────── Step 0 ──────────────────────────────────────────╮
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓         │
│ ┃ Prompt                            ┃ Completion                      ┃ Reward ┃         │
│ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩         │
│ │ system: You are a word counting   │ assistant: I'll count the words │   1.30 │         │
│ │ assistant. Count the number of    │ in the given text.              │        │         │
│ │ words in the given text and       │                                 │        │         │
│ │ provide your answer in the        │ The text contains the word      │        │         │
│ │ following format:                 │ "word" repeated multiple times. │        │         │
│ │                                   │ Let me count each occurrence:   │        │         │
│ │ <word_count>                      │                                 │        │         │
│ │ [number]                          │ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  │        │         │
│ │ </word_count>                     │ 11, 12, 13, 14, 15, 16, 17, 18, │        │         │
│ │                                   │ 19, 20, 21, 22, 23, 24, 25, 26, │        │         │
│ │ user: Count the number of words   │ 27, 28, 29, 30, 31, 32, 33, 34, │        │         │
│ │ in the following text:            │ 35, 36, 37, 38, 39, 40, 41, 42, │        │         │
│ │                                   │ 43, 44, 45                      │        │         │
│ │ word word word word word word ... │                                 │        │         │
│ │                                   │ <word_count>                    │        │         │
│ │                                   │ 45                              │        │         │
│ │                                   │ </word_count>                   │        │         │
│ └───────────────────────────────────┴─────────────────────────────────┴────────┘         │
╰────────────────────────────────────────────────────────────────────────────────────────────╯

--- All ---
Rewards:
reward: avg - 1.300, std - 0.000
r1: [1.3, 1.3, 1.3, 1.3, 1.3]

exact_match_reward: avg - 1.000, std - 0.000
r1: [1.0, 1.0, 1.0, 1.0, 1.0]

format_reward: avg - 1.000, std - 0.000
r1: [1.0, 1.0, 1.0, 1.0, 1.0]

partial_credit_reward: avg - 1.000, std - 0.000
r1: [1.0, 1.0, 1.0, 1.0, 1.0]
```

**Análise dos Resultados:**
- ✅ **Reward médio: 1.30** (máximo possível!)
- ✅ **100% de acerto** em todas as métricas
- ✅ Claude Sonnet contou corretamente todas as palavras
- ✅ Formatação XML perfeita em todos os casos
- ⚡ Avaliação completada em **2.36 segundos**

## 🔄 Geração de Rollouts para Treinamento RL

Depois de avaliar seu modelo, você pode gerar **rollouts** para treinar com o [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl).

### O que são Rollouts?

**Rollouts** são interações completas registradas que servem como dados de treinamento para RL:
- 📝 **Prompt**: A entrada do usuário
- 💬 **Completion**: A resposta do modelo
- ⭐ **Reward**: Pontuação calculada pelo rubric
- 📊 **Metadados**: Informações adicionais

### Geração Automática de Rollouts

O script `generate_rollouts.py` usa o Verifiers para gerar rollouts automaticamente:

```bash
# 1. Certifique-se que o LiteLLM está rodando
litellm --config litellm/litellm_config.yaml --port 8000

# 2. Gere rollouts
cd rollout_generation
uv run generate_rollouts.py
```

**Saída:**
```
🚀 Gerando 10 rollouts via http://localhost:8000
⚙️  Concorrência: 2 requisições simultâneas
⏳ Chamando modelo claude-sonnet...
Processing 10 groups (10 total rollouts): 100%|██████| 10/10 [00:13<00:00, 1.31s/it]

✅ Rollouts salvos em: rollouts/rollouts.jsonl
   Formato: JSONL (uma linha = um rollout)
   Total: 10 rollouts

📊 Estatísticas:
   Reward médio: 1.300
   Reward std: 0.000
   Min: 1.300
   Max: 1.300
```

### Formato JSONL

Cada linha do arquivo `rollouts.jsonl` contém um rollout completo:

```json
{
  "prompt": [
    {"role": "system", "content": "You are a word counting assistant..."},
    {"role": "user", "content": "Count the words: word word word..."}
  ],
  "completion": [
    {"role": "assistant", "content": "<word_count>\n45\n</word_count>"}
  ],
  "reward": 1.3,
  "answer": "45"
}
```

### Configuração

Edite as variáveis no final de `generate_rollouts.py`:

```python
NUM_EXAMPLES = 10          # Quantos rollouts gerar
MODEL = "claude-sonnet"    # Nome do modelo no litellm_config.yaml
BASE_URL = "http://localhost:8000"
MAX_CONCURRENT = 2         # Concorrência (ajuste se rate limit)
```

### Uso com Prime-RL

Os rollouts gerados estão prontos para treinar com Prime-RL:

```toml
# config.toml para Prime-RL
[model]
name = "meta-llama/Llama-3.1-8B-Instruct"

[environment]
env_id = "word-count"

[data]
rollout_dir = "./rollout_generation/rollouts"
rollout_files = ["rollouts.jsonl"]
batch_size = 32

[training]
algorithm = "ppo"
num_epochs = 3
learning_rate = 1e-5

[logging]
wandb_project = "word-count-rl"
```

Execute o treinamento:

```bash
# Instalar Prime-RL
pip install prime-rl

# Treinar modelo
uv run trainer @ config.toml
```

### Dicas para Geração de Rollouts

**Se encontrar Rate Limit (AWS Bedrock):**
- Reduza `NUM_EXAMPLES` (ex: 5)
- Reduza `MAX_CONCURRENT` (ex: 1)
- Gere em batches menores

**Para dados de qualidade variada:**
- Ajuste `temperature` do modelo para ter respostas mais diversas
- Use diferentes `seed` para variar os exemplos
- Misture rollouts de diferentes modelos

**Para produção contínua:**
- Use o Orchestrator do Prime-RL para coleta automática
- Implemente o middleware mostrado na seção "Implementação em Produção"
- Configure coleta periódica via cron ou serviço

## ⚙️ Outras Opções de API

### OpenAI (Direto)
```bash
export OPENAI_API_KEY=your_key
uv run vf-eval word-count -m gpt-4o-mini -n 10 -r 2
```

### Anthropic (Direto)
```bash
export ANTHROPIC_API_KEY=your_key
uv run vf-eval word-count -m claude-sonnet-4-20250514 -n 10 -r 2
```

### vLLM (Modelo Local)
```bash
# 1. Inicie o servidor vLLM
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# 2. Execute o eval
uv run vf-eval word-count \
  -m meta-llama/Llama-3.1-8B-Instruct \
  -b http://localhost:8000/v1 \
  -n 10 -r 2
```

## 🔧 Configuração do Ambiente

### Argumentos Customizáveis

Configure via linha de comando com `-a`:

```bash
uv run vf-eval word-count \
  -m claude-sonnet \
  -b http://localhost:8000 \
  -k LITELLM_API_KEY \
  -n 10 -r 2 \
  -a '{"num_examples": 50, "min_words": 3, "max_words": 20, "seed": 123}'
```

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `num_examples` | int | 100 | Número de exemplos a gerar |
| `min_words` | int | 5 | Mínimo de palavras por exemplo |
| `max_words` | int | 50 | Máximo de palavras por exemplo |
| `seed` | int | 42 | Seed para reprodutibilidade |

## 📊 Métricas Reportadas

O ambiente calcula e reporta as seguintes métricas em cada avaliação:

| Métrica | Descrição | Faixa |
|---------|-----------|-------|
| `reward` | Recompensa total ponderada (soma de todas as recompensas × pesos) | 0.0 - 1.3 |
| `exact_match_reward` | Resposta exata: 1.0 se correto, 0.0 se incorreto | 0.0 - 1.0 |
| `format_reward` | Uso correto das tags XML `<word_count>N</word_count>` | 0.0 - 1.0 |
| `partial_credit_reward` | Crédito parcial baseado na proximidade da resposta | 0.0 - 1.0 |

Resultados são automaticamente salvos em formato compatível com HuggingFace Datasets.

## 🎓 Usando como Template

Este ambiente pode servir como base para criar seus próprios ambientes:

### Passo a Passo

1. **Copie a estrutura** de `environments/word_count/`
2. **Modifique `load_environment()`** com sua lógica de dataset
3. **Ajuste o parser** conforme o formato de saída desejado
4. **Defina funções de recompensa** específicas para sua tarefa
5. **Atualize o `pyproject.toml`** com o novo nome e metadados
6. **Instale** com `uv pip install -e .`
7. **Teste** com `vf-eval seu-ambiente`

### Componentes Principais

```python
# environments/seu_ambiente/seu_ambiente.py
import verifiers as vf
from datasets import Dataset

def load_environment(**kwargs) -> vf.Environment:
    # 1. Crie ou carregue seu dataset
    dataset = Dataset.from_list([...])
    
    # 2. Defina o parser (XMLParser, ThinkParser, ou custom)
    parser = vf.XMLParser(["answer_field"])
    
    # 3. Defina funções de recompensa
    def reward_func(completion, answer, **kwargs) -> float:
        # Sua lógica aqui
        return score
    
    # 4. Crie o rubric
    rubric = vf.Rubric(
        funcs=[reward_func],
        weights=[1.0],
        parser=parser
    )
    
    # 5. Retorne o ambiente
    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        system_prompt="...",
        **kwargs
    )
```

## 🔗 Recursos Úteis

- **[Verifiers Framework](https://github.com/PrimeIntellect-ai/verifiers)** - Framework para ambientes e avaliação
- **[Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)** - Framework para treinamento RL em escala
- **[Documentação Verifiers](https://prime-rl.readthedocs.io/)** - Guias e referências
- **[Exemplos de Ambientes](https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments)** - Mais templates
- **[LiteLLM](https://docs.litellm.ai/)** - Documentação do proxy universal
- **[rollout_generation/README.md](./rollout_generation/README.md)** - Documentação detalhada de geração de rollouts

## 🚀 Implementação em Produção

### Arquitetura para Coleta de Dados de RL

Para usar este ambiente em produção e coletar dados para treinamento com Reinforcement Learning, você pode implementar um sistema de **middleware de avaliação** que intercepta todas as inferências do seu agente.

#### Arquitetura Proposta

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│   Cliente   │ ───> │  API Gateway     │ ───> │   Agente    │
│  (Usuário)  │      │  + Middleware    │      │    (LLM)    │
└─────────────┘      └──────────────────┘      └─────────────┘
                              │
                              ├─> Calcula Reward (Rubric)
                              │
                              ▼
                     ┌──────────────────┐
                     │   Banco de Dados │
                     │  (Rewards Store) │
                     └──────────────────┘
                              │
                              │ Batch Processing
                              ▼
                     ┌──────────────────┐
                     │  Prime Framework │
                     │  (RL Training)   │
                     └──────────────────┘
```

### Implementação Passo a Passo

#### 1. Middleware de Avaliação

Crie um middleware que intercepta as requisições/respostas do agente:

```python
# production/middleware.py
import asyncio
from datetime import datetime
from typing import Dict, Any
import verifiers as vf
from word_count import load_environment

class RewardMiddleware:
    def __init__(self):
        # Carregue o ambiente sem dataset (apenas rubric)
        self.env = load_environment(num_examples=0)
        self.rubric = self.env.rubric
        self.parser = self.env.parser
        
    async def evaluate_interaction(
        self,
        prompt: str,
        completion: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Avalia uma interação em tempo real.
        
        Args:
            prompt: A pergunta/prompt do usuário
            completion: A resposta do agente
            metadata: Metadados adicionais (user_id, session_id, etc)
        
        Returns:
            Dict com rewards e metadados
        """
        # Formato das mensagens para o rubric
        completion_msgs = [{"role": "assistant", "content": completion}]
        
        # Extrair resposta esperada (se disponível) ou usar heurísticas
        answer = self._extract_ground_truth(prompt, metadata)
        
        # Calcular rewards usando o rubric
        rewards = {}
        for func, weight in zip(self.rubric.funcs, self.rubric.weights):
            try:
                reward_value = func(
                    completion=completion_msgs,
                    answer=answer,
                    prompt=prompt,
                    info=metadata
                )
                rewards[func.__name__] = {
                    "value": reward_value,
                    "weight": weight,
                    "weighted_value": reward_value * weight
                }
            except Exception as e:
                rewards[func.__name__] = {
                    "error": str(e),
                    "value": 0.0,
                    "weight": weight
                }
        
        # Calcular reward total
        total_reward = sum(r["weighted_value"] for r in rewards.values())
        
        return {
            "prompt": prompt,
            "completion": completion,
            "rewards": rewards,
            "total_reward": total_reward,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _extract_ground_truth(self, prompt: str, metadata: Dict) -> str:
        """
        Extrai ou infere a resposta correta.
        Em produção, pode vir de metadados, validação humana, ou heurísticas.
        """
        return metadata.get("expected_answer", "")
```

#### 2. Integração com API

Integre o middleware na sua API principal:

```python
# production/api.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio

from middleware import RewardMiddleware
from storage import RewardStorage

app = FastAPI()
reward_middleware = RewardMiddleware()
storage = RewardStorage()

class InferenceRequest(BaseModel):
    prompt: str
    user_id: str
    session_id: str
    expected_answer: str = None  # Opcional: para validação

class InferenceResponse(BaseModel):
    completion: str
    reward_score: float = None
    interaction_id: str

@app.post("/inference", response_model=InferenceResponse)
async def inference(
    request: InferenceRequest,
    background_tasks: BackgroundTasks
):
    # 1. Fazer inferência no agente/LLM
    completion = await call_agent(request.prompt)
    
    # 2. Calcular reward em background (não bloqueia resposta)
    interaction_id = generate_id()
    
    background_tasks.add_task(
        evaluate_and_store,
        interaction_id=interaction_id,
        prompt=request.prompt,
        completion=completion,
        metadata={
            "user_id": request.user_id,
            "session_id": request.session_id,
            "expected_answer": request.expected_answer
        }
    )
    
    return InferenceResponse(
        completion=completion,
        interaction_id=interaction_id
    )

async def evaluate_and_store(
    interaction_id: str,
    prompt: str,
    completion: str,
    metadata: dict
):
    """Avalia e armazena rewards em background."""
    # Avaliar
    result = await reward_middleware.evaluate_interaction(
        prompt=prompt,
        completion=completion,
        metadata=metadata
    )
    
    # Armazenar
    await storage.store_interaction(
        interaction_id=interaction_id,
        data=result
    )
```

#### 3. Armazenamento de Dados

Configure um banco de dados para armazenar as interações e rewards:

```python
# production/storage.py
import asyncpg
from typing import Dict, Any
import json

class RewardStorage:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(self.db_url)
    
    async def store_interaction(
        self,
        interaction_id: str,
        data: Dict[str, Any]
    ):
        """Armazena interação e rewards no banco."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO rl_interactions (
                    interaction_id,
                    prompt,
                    completion,
                    total_reward,
                    rewards_detail,
                    metadata,
                    timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                interaction_id,
                data["prompt"],
                data["completion"],
                data["total_reward"],
                json.dumps(data["rewards"]),
                json.dumps(data["metadata"]),
                data["timestamp"]
            )
    
    async def get_training_batch(
        self,
        batch_size: int = 1000,
        min_reward: float = None
    ):
        """Recupera batch de dados para treinamento."""
        async with self.pool.acquire() as conn:
            query = """
                SELECT 
                    prompt,
                    completion,
                    total_reward,
                    rewards_detail,
                    metadata
                FROM rl_interactions
                WHERE processed = FALSE
            """
            if min_reward is not None:
                query += f" AND total_reward >= {min_reward}"
            
            query += f" LIMIT {batch_size}"
            
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]
```

#### 4. Schema do Banco de Dados

```sql
-- production/schema.sql
CREATE TABLE rl_interactions (
    interaction_id VARCHAR(255) PRIMARY KEY,
    prompt TEXT NOT NULL,
    completion TEXT NOT NULL,
    total_reward FLOAT NOT NULL,
    rewards_detail JSONB NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMP NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    used_in_training BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Índices para queries rápidas
CREATE INDEX idx_reward ON rl_interactions(total_reward);
CREATE INDEX idx_timestamp ON rl_interactions(timestamp);
CREATE INDEX idx_processed ON rl_interactions(processed);
CREATE INDEX idx_user_id ON rl_interactions((metadata->>'user_id'));
```

#### 5. Integração com Prime Framework

Script para exportar dados e treinar com o Prime Framework:

```python
# production/train_with_prime.py
import asyncio
from datasets import Dataset
import verifiers as vf
from storage import RewardStorage

async def export_to_prime_format(storage: RewardStorage, output_path: str):
    """Exporta dados coletados para formato Prime."""
    
    # 1. Recuperar dados do banco
    interactions = await storage.get_training_batch(
        batch_size=10000,
        min_reward=0.5  # Apenas interações com reward razoável
    )
    
    # 2. Converter para formato Verifiers
    dataset_dict = []
    for interaction in interactions:
        dataset_dict.append({
            "prompt": [{"role": "user", "content": interaction["prompt"]}],
            "completion": [{"role": "assistant", "content": interaction["completion"]}],
            "reward": interaction["total_reward"],
            "info": interaction["metadata"]
        })
    
    # 3. Criar Dataset HuggingFace
    dataset = Dataset.from_list(dataset_dict)
    
    # 4. Salvar
    dataset.save_to_disk(output_path)
    print(f"✅ Dataset exportado: {len(dataset)} exemplos em {output_path}")
    
    return dataset

async def train_with_prime():
    """Executa treinamento RL com Prime Framework."""
    storage = RewardStorage(db_url="postgresql://...")
    await storage.connect()
    
    # Exportar dados
    dataset = await export_to_prime_format(
        storage=storage,
        output_path="./training_data/word_count_prod"
    )
    
    # Carregar ambiente
    env = vf.load_environment("word-count")
    
    # Configurar e executar treinamento com Prime
    # (ver documentação do Prime Framework para detalhes)
    """
    from prime_rl import PRLConfig, train
    
    config = PRLConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        dataset=dataset,
        environment=env,
        num_epochs=3,
        batch_size=32,
        learning_rate=1e-5
    )
    
    trained_model = train(config)
    """

if __name__ == "__main__":
    asyncio.run(train_with_prime())
```

### Fluxo de Produção Completo

1. **Coleta em Tempo Real**
   - Usuário faz pergunta → API processa
   - Agente gera resposta → Enviada ao usuário
   - Middleware calcula reward em background
   - Dados armazenados no banco

2. **Análise e Filtro**
   - Dashboard para monitorar distribuição de rewards
   - Identificar casos de baixo reward para revisão
   - Filtrar dados de qualidade para treinamento

3. **Preparação para Treinamento**
   - Exportar batch de interações (ex: últimos 7 dias)
   - Balancear dataset (diferentes faixas de reward)
   - Converter para formato HuggingFace Dataset

4. **Treinamento com Prime**
   - Usar dados coletados como experiências
   - Treinar modelo com PPO/GRPO
   - Avaliar modelo melhorado
   - Deploy gradual (A/B testing)

### Considerações de Produção

#### Performance
- **Rewards em background**: Não bloqueie a resposta ao usuário
- **Cache**: Use Redis para rubrics frequentes
- **Batch processing**: Processe múltiplas avaliações em paralelo

#### Privacidade e Segurança
- **Anonimização**: Remova PII antes de armazenar
- **Retenção**: Defina políticas de retenção de dados
- **Auditoria**: Logs de todas as avaliações

#### Monitoramento
```python
# Métricas importantes
- Distribuição de rewards (histograma)
- Taxa de interações com reward > threshold
- Latência do middleware
- Volume de dados coletados por dia
```

#### Qualidade dos Dados
- **Validação humana**: Amostras aleatórias revisadas por humanos
- **Ground truth**: Para tarefas com resposta certa, valide contra ela
- **Feedback do usuário**: Capture thumbs up/down para ajustar rewards

### Exemplo de Configuração Docker

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: ./production
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/rl_data
    ports:
      - "8000:8000"
    depends_on:
      - db
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=rl_data
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  training_worker:
    build: ./production
    command: python train_with_prime.py
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/rl_data
    depends_on:
      - db

volumes:
  postgres_data:
```

Este setup permite coletar dados de produção continuamente e retreinar seu modelo periodicamente, criando um **ciclo de melhoria contínua** com Reinforcement Learning.

## 💡 Dicas e Solução de Problemas

### Erros Comuns

**Erro: "No module named 'word_count'"**
```bash
cd environments/word_count && uv pip install -e .
```

**Erro: "Connection refused" no LiteLLM**
```bash
# Verifique se o proxy está rodando
litellm --config litellm/litellm_config.yaml --port 8000
```

**Erro: AWS Credentials**
```bash
# Configure as credenciais AWS
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### Performance

- Use `-n` menor para testes rápidos (ex: `-n 5`)
- Aumente `max_concurrent` para paralelismo maior
- Use `temperature=0` para resultados determinísticos

## 📝 Licença

MIT License

## 🤝 Contribuindo

Contribuições são bem-vindas! Este projeto é um exemplo educacional.

---

## 📌 Status do Projeto

| Componente | Status | Detalhes |
|------------|--------|----------|
| **Ambiente** | ✅ Pronto | word_count environment implementado e testado |
| **Avaliação** | ✅ Pronto | vf-eval funcionando com LiteLLM + AWS Bedrock |
| **Geração de Rollouts** | ✅ Pronto | Script automático gerando JSONL para Prime-RL |
| **Treinamento RL** | 📋 Documentado | Guia completo para uso com Prime-RL |
| **Produção** | 📋 Arquitetura | Middleware e pipeline documentados |

**Versão:** 0.1.0  
**Frameworks:** Verifiers >= 0.1.8.post2 | Prime-RL (compatível)  
**Última atualização:** Dezembro 2025