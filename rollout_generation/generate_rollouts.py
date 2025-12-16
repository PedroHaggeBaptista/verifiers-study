"""
Gera rollouts usando o ambiente word_count via LiteLLM.

Uso:
1. Inicie o LiteLLM proxy: litellm --config ../litellm/litellm_config.yaml --port 8000
2. Execute: python generate_rollouts.py
"""
import asyncio
import sys
from pathlib import Path

# Adicionar o ambiente ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "environments" / "word_count"))

from word_count import load_environment
from openai import AsyncOpenAI


async def generate_rollouts(
    num_examples: int = 10,
    model: str = "claude-sonnet",
    base_url: str = "http://localhost:8000",
    max_concurrent: int = 3  # Reduzir concorrência para evitar rate limit
):
    """
    Gera rollouts usando LiteLLM.
    
    Args:
        num_examples: Quantos exemplos gerar
        model: Nome do modelo no litellm_config.yaml
        base_url: URL do proxy LiteLLM
        max_concurrent: Máximo de requisições simultâneas (reduzir se hit rate limit)
    """
    print(f"🚀 Gerando {num_examples} rollouts via {base_url}")
    print(f"⚙️  Concorrência: {max_concurrent} requisições simultâneas")
    
    # 1. Carregar ambiente
    env = load_environment(
        num_examples=num_examples,
        min_words=5,
        max_words=50,
        seed=42
    )
    
    # 2. Configurar cliente OpenAI apontando para LiteLLM
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="dummy",  # LiteLLM não precisa de key real
        timeout=60.0,  # Timeout maior
        max_retries=3  # Retry automático
    )
    
    # 3. Gerar rollouts com concorrência controlada
    print(f"⏳ Chamando modelo {model}...")
    
    results = await env.evaluate(
        client=client,
        model=model,
        num_examples=num_examples,
        rollouts_per_example=1,
        max_concurrent=max_concurrent  # Limitar concorrência
    )
    
    # 4. Salvar rollouts como JSONL (formato padrão para Prime-RL)
    import json
    output_dir = Path(__file__).parent / "rollouts"
    output_dir.mkdir(exist_ok=True)
    
    jsonl_file = output_dir / "rollouts.jsonl"
    
    # Converter cada rollout para JSON e salvar uma linha por vez
    num_rollouts = len(results['reward'])
    with open(jsonl_file, "w") as f:
        for i in range(num_rollouts):
            rollout = {
                "prompt": results['prompt'][i],
                "completion": results['completion'][i],
                "reward": float(results['reward'][i]),
                "answer": results['answer'][i],
            }
            # Adicionar outras métricas
            for key in ['exact_match_reward', 'format_reward', 'partial_credit_reward']:
                if key in results:
                    rollout[key] = float(results[key][i])
            
            f.write(json.dumps(rollout) + "\n")
    
    print(f"\n✅ Rollouts salvos em: {jsonl_file}")
    print(f"   Formato: JSONL (uma linha = um rollout)")
    
    # 5. Mostrar estatísticas
    print(f"   Total: {num_rollouts} rollouts")
    
    import numpy as np
    rewards = np.array(results['reward'])
    
    print(f"\n📊 Estatísticas:")
    print(f"   Reward médio: {rewards.mean():.3f}")
    print(f"   Reward std: {rewards.std():.3f}")
    print(f"   Min: {rewards.min():.3f}")
    print(f"   Max: {rewards.max():.3f}")
    
    return results


if __name__ == "__main__":
    # Configuração simples
    NUM_EXAMPLES = 10  # Reduzido para evitar rate limit
    MODEL = "claude-sonnet"  # Nome do modelo no litellm_config.yaml
    BASE_URL = "http://localhost:8000"  # URL do LiteLLM
    MAX_CONCURRENT = 2  # Apenas 2 requisições simultâneas (Bedrock tem limite baixo)
    
    asyncio.run(generate_rollouts(
        num_examples=NUM_EXAMPLES,
        model=MODEL,
        base_url=BASE_URL,
        max_concurrent=MAX_CONCURRENT
    ))

