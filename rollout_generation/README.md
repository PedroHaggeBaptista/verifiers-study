# Geração de Rollouts para Prime-RL

Script simples para gerar rollouts usando o ambiente `word_count` via LiteLLM.

## 📋 O que são Rollouts?

**Rollouts** são execuções completas de interações que incluem:
- **Prompt**: A entrada do usuário
- **Completion**: A resposta do modelo  
- **Reward**: Pontuação calculada pelo rubric

Esses rollouts são usados para treinar modelos com RL no Prime-RL.

## 🚀 Como Usar

### 1. Iniciar LiteLLM

```bash
# Na pasta raiz do projeto
litellm --config litellm/litellm_config.yaml --port 8000
```

### 2. Gerar Rollouts

```bash
cd rollout_generation
python generate_rollouts.py
```

### 3. Configurar (opcional)

Edite as variáveis no final de `generate_rollouts.py`:

```python
NUM_EXAMPLES = 20           # Quantos rollouts gerar
MODEL = "claude-sonnet"     # Nome no litellm_config.yaml
BASE_URL = "http://localhost:8000"  # URL do LiteLLM
```

## 📊 Output

Após executar, você terá:

```
rollout_generation/
└── rollouts/              # Dataset HuggingFace
    ├── dataset_info.json
    └── data-00000-of-00001.arrow
```

**Exemplo de saída:**
```
🚀 Gerando 20 rollouts via http://localhost:8000
⏳ Chamando modelo claude-sonnet...

✅ 20 rollouts salvos em: rollouts/

📊 Estatísticas:
   Reward médio: 1.250
   Reward std: 0.089
   Min: 1.100
   Max: 1.300

📝 Exemplo de rollout:
   Prompt: Count the number of words in the following text: word word word...
   Completion: <word_count> 45 </word_count>
   Reward: 1.30
```

## 📊 Estrutura dos Rollouts Gerados

Cada rollout tem a seguinte estrutura:

```python
{
    "prompt": [
        {"role": "user", "content": "Count the words: hello world"}
    ],
    "completion": [
        {"role": "assistant", "content": "<word_count>\n2\n</word_count>"}
    ],
    "reward": 1.3,                    # Reward total ponderado
    "exact_match_reward": 1.0,        # Reward individual
    "format_reward": 1.0,             # Reward individual
    "partial_credit_reward": 1.0,     # Reward individual
    "answer": "2",                    # Ground truth
    "info": {"text": "hello world", "word_count": 2},
    "metadata": {"user_id": "...", "session_id": "..."}
}
```

## 🔗 Integração com Prime-RL

Depois de gerar os rollouts, use com Prime-RL:

### 1. Criar configuração de treinamento

```toml
# configs/train_with_rollouts.toml

[model]
name = "meta-llama/Llama-3.1-8B-Instruct"

[environment]
env_id = "word-count"

[data]
# Aponta para o diretório de rollouts gerados
rollout_dir = "./rollout_generation/rollouts_custom"
batch_size = 32

[training]
algorithm = "ppo"
num_epochs = 3
learning_rate = 1e-5

[logging]
wandb_project = "word-count-rl"
```

### 2. Executar treinamento

```bash
# Instalar Prime-RL (se ainda não instalou)
pip install prime-rl

# Treinar usando os rollouts gerados
uv run trainer @ configs/train_with_rollouts.toml
```

## 📈 Análise de Rollouts

### Inspecionar rollouts gerados:

```python
from datasets import load_from_disk

# Carregar dataset
dataset = load_from_disk("./rollouts_custom")

# Ver estatísticas
print(f"Total rollouts: {len(dataset)}")
print(f"Reward médio: {sum(dataset['reward']) / len(dataset):.3f}")
print(f"Colunas: {dataset.column_names}")

# Ver exemplos
for i in range(3):
    print(f"\nRollout {i}:")
    print(f"  Prompt: {dataset[i]['prompt']}")
    print(f"  Completion: {dataset[i]['completion']}")
    print(f"  Reward: {dataset[i]['reward']:.2f}")
```

### Filtrar rollouts por qualidade:

```python
# Manter apenas rollouts com reward > 1.0
high_quality = dataset.filter(lambda x: x['reward'] > 1.0)
high_quality.save_to_disk("./rollouts_filtered")

print(f"Filtrados: {len(high_quality)}/{len(dataset)} rollouts")
```

## 🎯 Cenários de Uso

### Desenvolvimento e Debug
```bash
# Gerar poucos rollouts mock para testes rápidos
# Modifique num_examples=10 para num_examples=100
python generate_rollouts.py
```

### Avaliação de Modelo
```bash
# Gerar rollouts com modelo atual para baseline
export API_BASE_URL=http://seu-modelo:8000
python generate_rollouts.py
# Compara rewards antes/depois do treinamento
```

### Produção Contínua
```bash
# Agendar geração periódica (via cron)
0 */6 * * * cd /path/to/rollout_generation && python generate_rollouts.py

# Ou usar um serviço que monitora seu banco de dados
# e gera rollouts automaticamente quando há dados novos
```

## 📁 Estrutura de Arquivos

```
rollout_generation/
├── generate_rollouts.py       # Script principal
├── README.md                  # Este arquivo
├── rollouts_mock/            # Rollouts simulados
│   ├── dataset_info.json
│   └── data-00000-of-00001.arrow
├── rollouts_api/             # Rollouts via API
│   └── ...
├── rollouts_custom/          # Rollouts customizados
│   └── ...
└── rollouts_custom.jsonl     # Formato JSONL alternativo
```

## 🔍 Troubleshooting

**Erro: "No module named 'word_count'"**
```bash
cd ../environments/word_count
uv pip install -e .
```

**Erro: API não responde**
```bash
# Verificar se servidor está rodando
curl http://localhost:8000/health

# Verificar logs do LiteLLM
litellm --config ../litellm/litellm_config.yaml --debug
```

**Rollouts com reward muito baixo**
- Verifique se o formato da resposta está correto
- Inspecione o campo `completion` dos rollouts
- Ajuste o system prompt do modelo se necessário

## 📚 Próximos Passos

1. ✅ Gerar rollouts com este script
2. ✅ Inspecionar e validar a qualidade
3. ✅ Configurar Prime-RL para usar os rollouts
4. ✅ Treinar modelo com RL
5. ✅ Avaliar modelo treinado vs baseline
6. ✅ Iterar: coletar mais dados → treinar → avaliar

---

**Dúvidas?** Consulte o README principal do projeto ou a documentação do Prime-RL.

