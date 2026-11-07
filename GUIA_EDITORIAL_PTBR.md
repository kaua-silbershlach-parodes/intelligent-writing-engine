# Guia editorial de inteligência — português brasileiro

> **Status:** diretriz editorial para revisão de conteúdo. Não é uma validação de informações, de fontes, de técnicas ou de produtos. As “fontes redigidas” são uma proposta futura; sua implementação não é presumida.

## Objetivo

Produzir textos claros, úteis e verificáveis para o público da A Elite Estrangeira. A redação deve apresentar o ponto principal sem ambiguidades, manter linguagem consistente em português brasileiro e distinguir fatos de interpretações.

## Seis princípios

1. **Clareza:** apresente a informação de modo direto. Use termos precisos, frases legíveis e definições quando houver siglas pouco conhecidas. Cada palavra deve contribuir para o significado.
2. **Concisão:** elimine redundâncias, abreviações confusas, frases vagas e palavras dispensáveis. Não sacrifique precisão para encurtar o texto.
3. **Relevância — “E daí?”:** explicite por que o assunto importa, qual é seu impacto e o que o leitor precisa saber ou decidir. Remova conteúdo sem utilidade para o objetivo.
4. **Análise do público:** adapte vocabulário, nível de detalhe e contexto às necessidades do público principal e secundário. Não presuma conhecimento técnico universal.
5. **Além da observação:** diferencie relato de fatos, análise e avaliação prospectiva. Uma observação isolada não constitui uma conclusão analítica.
6. **Objetividade:** examine pressupostos, possíveis vieses e explicações alternativas. Solicite revisão independente quando o contexto permitir. Não transforme possibilidade em certeza.

## Estrutura recomendada para análises

- **BLUF (conclusão primeiro):** apresente o julgamento principal logo no início quando houver base para uma conclusão.
- **Evidências:** identifique fatos e fontes que sustentam cada afirmação material. Preserve origem, data e limitações de cada fonte.
- **Avaliação:** destaque separadamente interpretações, hipóteses, incertezas e grau de confiança, sem inventar probabilidades ou certezas.
- **Implicações:** explique o significado prático para o público e para a decisão em questão.
- **Alternativas:** quando relevante, considere hipóteses concorrentes e informações que poderiam contrariar a conclusão.
- **Revisão:** verifique coerência, ortografia, pontuação, atribuição de fontes e potencial de viés.

## Padrão para interfaces e footer

1. Escreva os rótulos em **português brasileiro natural**, usando preferencialmente substantivos ou expressões nominais curtas em grupos de navegação.
2. Evite traduções literais e abreviações sem contexto, como “Rotas Alertas”, “Revisão Fonte”, “Política Priv.” ou “Mod. Diamante”.
3. Preserve siglas técnicas reconhecidas, como CTI, DNS, API, IOC, STIX/TAXII, XSS e CSRF. Explique-as em textos de apoio quando necessário.
4. Use termos consistentes: “vazamento de dados”, “indicadores de comprometimento (IOCs)”, “atores de ameaças”, “superfície de ataque”, “inteligência de ameaças”.
5. Não invente serviços, disponibilidade, preços, resultados, aprovações ou vínculos institucionais. Os rótulos não devem prometer funcionalidades inexistentes.
6. Preserve a estrutura, os destinos dos links, a acessibilidade e o tema visual ao editar conteúdo de uma página.
7. Faça a correção no **HTML de origem**. Não dependa de JavaScript executado no navegador para traduzir textos estáticos.
8. Verifique capitalização, concordância, regência, acentuação, tamanho das frases e consistência terminológica antes da publicação.

### Exemplos de correção de rótulos

| Antes | Redação recomendada |
|---|---|
| Busca IOCs | Pesquisa de IOCs |
| Perfis de Ator | Perfis de atores de ameaças |
| Riscos Terceiros | Riscos de terceiros |
| Rotas Alertas | Encaminhamento de alertas |
| Revisão Fonte | Avaliação de fontes |
| Mod. Diamante | Modelo Diamante |
| Política Priv. | Política de privacidade |
| Gestão Acessos | Controle de acesso |
| Atores Ameaça | Atores de ameaças |
| Vazamentos Senha | Senhas expostas |
| Máfia de Resgate | Grupos de ransomware |
| Notas Versões | Notas de versão |

## Critérios para futuras “fontes redigidas”

Uma futura área de fontes redigidas deve exigir, no mínimo: título objetivo; natureza do registro (fonte, observação, análise ou avaliação); identificação e data da fonte; contexto; conteúdo essencial; relevância; limitações e incertezas; indicação explícita de interpretações; revisão editorial e rastreabilidade das alterações.

**Não classificar automaticamente uma fonte como verificada** apenas por ter recebido revisão linguística. Qualidade de redação e validade probatória são verificações diferentes.

## Limites e referências

Este guia adapta os seis princípios enviados pelo responsável pelo projeto e os temas enumerados no [README do intelligent-writing-engine](./README.md), incluindo BLUF, atenção a vieses, separação entre julgamento e evidência e avaliação de hipóteses alternativas.

O README menciona GADINTEL, West Point, CIA/OTE e Mercyhurst/MCIIS como referências históricas de técnicas de redação; não há, neste repositório, os respectivos textos integrais nem comprovação específica de que este guia reproduza suas regras completas. Verifique a fonte primária antes de atribuir uma regra a qualquer dessas instituições.
