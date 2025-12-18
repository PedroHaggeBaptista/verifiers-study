# Adaptive Lying Oracle - RL Challenge

Um ambiente de Reinforcement Learning onde um agente deve adaptar sua estratégia quando um oráculo muda seu comportamento de verdadeiro para mentiroso.

## 🎯 O Desafio

### Descrição

Imagine um jogo onde você precisa adivinhar um número secreto entre 1 e 100, mas só pode fazer perguntas do tipo "o número é maior que k?". O oráculo responde, mas tem um twist: **ele começa dizendo a verdade, mas depois de 200 perguntas, ele inverte todas as respostas!**

### Especificações

1. **Hidden Number**: Um número aleatório entre 1-100
2. **Ação do Agente**: Escolher `k` (1-100) e perguntar "o número é > k?"
3. **Comportamento do Oráculo**:
   - `t < 200`: Responde **verdade** (truthful)
   - `t ≥ 200`: Responde **mentira** (inverte a resposta)
4. **Recompensas**:
   - `+1.0` se `k == hidden_number` (acertou! episódio termina)
   - `-0.01` caso contrário (penalidade por não acertar)
5. **Duração**: Máximo de 500 passos por episódio

### O Objetivo

Criar um agente que:
- ✅ Começa com busca binária (assumindo oráculo verdadeiro)
- ✅ Detecta quando o oráculo começa a mentir
- ✅ Adapta sua estratégia (inverte interpretação das respostas)
- ✅ Encontra o número secreto mesmo com o oráculo mentiroso

## 🏗️ Arquitetura

### 1. Ambiente: `LyingOracleEnv`

```python
class LyingOracleEnv(vf.MultiTurnEnv):
    """
    Ambiente multi-turn que simula o oráculo adaptativo.
    
    - Oracle responde verdade para t < 200
    - Oracle inverte resposta para t >= 200
    - Episódio termina em 500 passos ou quando acerta
    """
```

**Características**:
- Herda de `MultiTurnEnv` do framework Verifiers
- State tracking: turno atual, resposta do oráculo, histórico
- Reward: +1 para acerto, -0.01 para erro
- Terminação: acerto ou max_turns

### 2. Agente: `AdaptiveAgent`

```python
class AdaptiveAgent:
    """
    Agente que detecta mudança de comportamento e adapta.
    
    Estratégias de Detecção:
    1. Contradições (bounds cruzam)
    2. Queda no reward médio
    3. Não convergência após muitos passos
    """
```

**Mecanismo de Adaptação**:

1. **Fase Inicial (t < 200)**:
   - Busca binária normal
   - Reduz espaço baseado em respostas verdadeiras
   - Converge rapidamente

2. **Detecção (t ≈ 200-220)**:
   - Detecta contradições: `low > high`
   - Rastreia queda no reward médio móvel
   - Contador de contradições atinge threshold

3. **Fase Adaptada (t > 220)**:
   - Switch: `oracle_lying = True`
   - Inverte interpretação: `effective_response = not oracle_response`
   - Busca binária volta a funcionar
   - Encontra o número

### 3. Notebook: `demo.ipynb`

Demonstração completa com:
- ✅ Execução de episódios
- ✅ Plots de rewards (individual + rolling mean)
- ✅ Visualização de cumulative reward
- ✅ Estado de adaptação do agente
- ✅ Estatísticas de performance

## 📊 Visualizações

O notebook gera 4 plots principais:

1. **Individual Rewards**: Mostra cada reward por turno
2. **Rolling Mean Reward**: 
   - ⬇️ **Drop em t=200** (oráculo começa a mentir)
   - ⬆️ **Recovery após adaptação** (agente detecta e corrige)
3. **Cumulative Reward**: Tendência geral ao longo do tempo
4. **Agent State**: 
   - Contradições acumuladas
   - Momento em que agente acredita que oráculo está mentindo

### Exemplo de Resultado

```
🎯 Hidden number: 73
⚙️ Oracle switches to lying mode at t=200

🔄 AGENT ADAPTATION: Detected lying oracle at turn 215!
✅ Found hidden number 73 at turn 238!

📊 Results:
  - Found: True
  - Turns taken: 238
  - Final cumulative reward: -1.37

📈 Performance Statistics:

Phase 1 - Truthful Oracle (t < 200):
  Average reward: -0.0100
  Steps: 200

Phase 2 - Oracle Lying, Agent Unaware (200 ≤ t < 215):
  Average reward: -0.0100
  Steps: 15
  Contradictions accumulated: 3

Phase 3 - Agent Adapted (t ≥ 215):
  Average reward: 0.0435  (recovery!)
  Steps: 24
  Found solution: True
```

## 🚀 Como Usar

### 1. Instalação

```bash
cd environments/lying_oracle
uv pip install -e .
```

### 2. Executar Testes

```bash
uv run test_environment.py
```

**Saída esperada**:
```
Running Lying Oracle Environment Tests...

✅ Environment created successfully!
✅ Dataset structure is correct!
✅ Oracle behavior correct! Hidden number: 45
✅ Reward calculation correct!
✅ Episode termination correct!
✅ Agent found 73 in 7 turns with binary search!
✅ Agent adapted at turn 215 (oracle started lying at 200)
✅ Full episode completed! Found 67 in 243 turns

🎉 All tests passed!
```

### 3. Executar Notebook

```bash
cd environments/lying_oracle
jupyter notebook demo.ipynb
```

Ou use o VS Code / Cursor para executar o notebook interativamente.

### 4. Usar Programaticamente

```python
from lying_oracle import load_environment
from agent import run_episode

# Carregar ambiente
env = load_environment(num_examples=10, max_turns=500, lying_threshold=200)

# Executar um episódio
hidden_number = 42
history, found, turns = run_episode(
    hidden_number=hidden_number,
    max_turns=500,
    lying_threshold=200,
    verbose=True
)

print(f"Success: {found}, Turns: {turns}")
```

## 🧠 Algoritmo do Agente

### Pseudocódigo

```python
def adaptive_binary_search(oracle, hidden_number):
    low, high = 1, 100
    oracle_lying = False
    contradictions = 0
    
    for turn in range(500):
        # Escolher k (meio do intervalo atual)
        k = (low + high) // 2
        
        # Obter resposta do oráculo
        response = oracle.ask(k, turn)
        
        # Se acredita que está mentindo, inverte
        if oracle_lying:
            response = not response
        
        # Atualizar bounds
        if response:  # hidden > k
            low = k + 1
        else:         # hidden <= k
            high = k
        
        # Detectar contradição
        if low > high:
            contradictions += 1
            
        # Switch para modo mentira
        if contradictions >= 3 and not oracle_lying:
            oracle_lying = True
            low, high = 1, 100  # Reset bounds
        
        # Verificar se acertou
        if k == hidden_number:
            return True, turn
    
    return False, 500
```

### Estratégias de Detecção

| Estratégia | Threshold | Descrição |
|------------|-----------|-----------|
| **Contradições** | ≥ 3 | Bounds cruzam (low > high) |
| **Reward Drop** | < -0.15 | Média móvel cai muito |
| **Não Convergência** | t > 180 e range > 30 | Não está convergindo |

## 📈 Resultados Esperados

### Métricas de Sucesso

Com 5 episódios:
- **Taxa de Sucesso**: ~100%
- **Turnos Médios**: ~250-300
- **Delay de Adaptação**: ~10-20 turnos após t=200

### Comportamento dos Plots

**Plot 1 - Rolling Mean**:
```
 0.00 |━━━━━━━━━━━━━━━━━━━━━
      |                    \
-0.01 |                     \_____ (drop at t=200)
      |                          /
      |                         /
      |                        /_____ (recovery after adaptation)
      |━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      0       100      200      300      400      500
```

**Plot 2 - Cumulative Reward**:
```
      |    /
  0.0 |   /
      |  /
      | /  \
 -2.0 |/    \___  (slope changes at t=200)
      |         \_____/\  (recovers after adaptation)
      |                \_____
      |━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      0       100      200      300      400      500
```

## 🔬 Experimentos Adicionais

### 1. Variar o Threshold de Mentira

```python
# Oracle começa a mentir mais cedo
history, found, turns = run_episode(
    hidden_number=42,
    lying_threshold=100,  # Ao invés de 200
    max_turns=500
)
```

### 2. Ajustar Sensibilidade de Detecção

```python
agent = AdaptiveAgent(
    detection_window=10,        # Janela menor (mais sensível)
    contradiction_threshold=2,  # Menos contradições necessárias
    reward_drop_threshold=-0.10 # Threshold menor
)
```

### 3. Testar com Diferentes Hidden Numbers

```python
# Números extremos
for hidden in [1, 50, 100]:
    history, found, turns = run_episode(hidden_number=hidden)
    print(f"Hidden={hidden}: Found in {turns} turns")
```

## 🎓 Integração com Verifiers Framework

Este ambiente é compatível com o framework Verifiers para:

### Avaliação com LLMs

```bash
# Usar com modelos via API
uv run vf-eval lying-oracle -m gpt-4o-mini -n 10 -r 1
```

### Geração de Rollouts

```python
from verifiers import load_environment, generate_rollouts

env = load_environment("lying-oracle")
rollouts = generate_rollouts(
    env=env,
    model="claude-sonnet",
    num_examples=100
)
```

### Treinamento RL

```python
from prime_rl import PRLConfig, train

config = PRLConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    environment="lying-oracle",
    num_epochs=3,
    batch_size=32
)

trained_model = train(config)
```

## 🧪 Desafios Avançados

### Nível 2: Oracle Aleatório

Modifique o oráculo para mentir **aleatoriamente** com probabilidade crescente:

```python
lie_probability = min(0.5, turn / 400)
if random.random() < lie_probability:
    response = not response
```

### Nível 3: Oracle com Padrões

Oracle mente em padrões específicos:
- A cada 3 perguntas
- Apenas para números ímpares
- Com probabilidade baseada no valor de k

### Nível 4: Multi-Agent

Múltiplos agentes competindo para encontrar o número primeiro, compartilhando (ou não) informações.

## 📚 Conceitos de RL Demonstrados

Este projeto demonstra:

1. **Online Learning**: Agente aprende durante execução
2. **Concept Drift**: Comportamento do ambiente muda ao longo do tempo
3. **Exploration vs Exploitation**: Busca binária (exploitation) vs adaptação (exploration)
4. **Reward Shaping**: -0.01 incentiva eficiência
5. **State Tracking**: Manter histórico para detecção de padrões
6. **Adaptive Behavior**: Mudar estratégia baseado em feedback

## 🔗 Recursos

- **[Verifiers Framework](https://github.com/PrimeIntellect-ai/verifiers)**
- **[Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)**
- **[Documentação Verifiers](https://prime-rl.readthedocs.io/)**

## 📝 Licença

MIT License

## 🤝 Contribuindo

Melhorias e extensões são bem-vindas! Algumas ideias:

- [ ] Implementar agente com Deep RL (DQN, PPO)
- [ ] Adicionar diferentes estratégias de detecção
- [ ] Criar visualizações interativas
- [ ] Benchmark com diferentes algoritmos
- [ ] Extender para múltiplos oráculos

---

**Versão**: 0.1.0  
**Autor**: Challenge implementation for RL + Verifiers  
**Data**: Dezembro 2025


