# 🧪 Experimento: Detecção Simplificada

## 🎯 Objetivo

Testar se uma detecção **SIMPLES** (apenas contradiction rate) funciona melhor que a detecção **COMPLEXA** (3 sinais combinados) no cenário do Hard Challenge.

## 📊 Configuração

### **Versão Original (lying_oracle_hard):**
```python
# Multi-signal detection:
lying_confidence = (
    0.4 × contradiction_signal +
    0.3 × reward_signal +
    0.3 × convergence_signal
)

# Thresholds:
trust_to_distrust = 0.7
min_observations = 20
```

### **Versão Simplificada (este diretório):**
```python
# Single-signal detection (como o desafio sugere!):
lying_confidence = contradiction_rate

# Thresholds ajustados:
trust_to_distrust = 0.5  # Mais sensível
min_observations = 15    # Detecta mais cedo
```

## 🔬 Hipótese

**Hipótese:** A detecção simplificada vai funcionar MELHOR porque:

1. **Contradiction rate é o único sinal confiável** no cenário pós-descoberta
2. **Reward/convergence signals diluem** o sinal útil
3. **Threshold mais baixo (0.5)** é mais apropriado para lying 80%
4. **Menos smoothing** = resposta mais rápida

## 📈 Resultados Esperados

### **Se a hipótese estiver correta:**
- ✅ Lying confidence sobe para > 0.5
- ✅ Agent muda para DISTRUST
- ✅ Vemos recovery (mesmo que parcial)

### **Se a hipótese estiver errada:**
- ❌ Lying confidence continua baixa
- ❌ Agent não adapta
- ❌ Resultado igual ao original

## 🚀 Como Rodar

```bash
cd lying_oracle_hard_test
source .venv/bin/activate
jupyter notebook demo_simplified.ipynb
```

## 📊 Comparação

| Métrica | Original | Simplificado |
|---------|----------|--------------|
| **Sinais** | 3 (contradiction + reward + convergence) | 1 (contradiction only) |
| **Threshold** | 0.7 | 0.5 |
| **Min obs** | 20 | 15 |
| **Smoothing** | EMA (alpha=0.3) | None (direto) |
| **Confidence máx** | ~0.15 | ??? |
| **Adaptou?** | Não | ??? |

## 💡 Lição

Este experimento testa o princípio:

> **"Keep It Simple, Stupid"**
> 
> Às vezes, over-engineering prejudica mais do que ajuda.
> O desafio sugeriu "a simple statistic" por uma razão!

Se funcionar melhor, prova que:
- ✅ Simplicidade > Complexidade
- ✅ Entender o contexto > Seguir fórmulas
- ✅ Um sinal forte > Três sinais fracos

