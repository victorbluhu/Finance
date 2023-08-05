Repositório de dados e projetos de Finanças

# Tabela de Conteúdos
1. [Introdução](#introdução)
2. [Yield Curve](#yield-curve)
3. [Factor Analysis](#factor-analysis)
  1. [Choques de Política Monetária](#choques-de-política-monetária)
     1. [Estimação por OLS](#estimação-por-ols)
     2. [Identificação via Event-Study](#identificação-via-event-study)
     3. [Identificação por Surpresas de Juros](#identificação-por-surpresas-de-juros)
     4. [Identificação por Heteroscedasticidade](#identificação-por-heteroscedasticidade)
     5. [Comentários sobre os Resultados](#comentários-sobre-os-resultados)
     6. [Próximos Passos](#próximos-passos)
  2. [Atribuição de Performance](#atribuição-de-performance)
4. [Trading Strategy](#trading-strategy)

# Introdução
Neste repositório eu concentro alguns estudos realizados (ou suas etapas preliminares).

A parte mais avançada hoje é a modelagem de curva de juros. A partir das classes _YieldCurve_, que concentra os métodos básicos para transformar e visualizar curvas de juros, e _AffineYieldCurveModel_, que concentra os métodos básicos para modelos com SDF e curva lineares nos mesmos estados, implemento o modelo de [Adrian, Crump and Moench (2013)](https://doi.org/10.1016/j.jfineco.2013.04.009) para decompor as taxas de juros nominal observadas na curva US (ou de qualquer outro país de interesse) entre a expectativa risco-neutra da rolagem das taxas curtas e o prêmio de risco cobrado para cada horizonte.
  Uma extensão que estou desenvolvendo à frente é a de decompor conjuntamente a curva de juros nominais e a curva de juros reais, permitindo criar também uma decomposição da inflação implícita (breakeven).

Na seção seguinte, [Factor Analysis](#factor-analysis), há alguns projetos que utilizam fatores de risco do [Nefin](https://nefin.com.br/data/risk_factors.html) tanto para projetar os prêmios de risco embutidos em outros ativos, quanto como portifólios de equities nos quais buscamos identificar efeitos de política monetária.
  Um projeto adicional a ser desenvolvido é o de [atribuição de performance a índices setoriais e fundos](#atribuição-de-performance) (cálculo de alphas e betas). Para este projeto, falta basicamente levantar um histórico de performance de fundos de investimento, mas ainda não fui atrás de uma fonte adequada para lidar com viés de sobrevivência. 


# Yield Curve
...

# Factor Analysis

## Choques de Política Monetária
Uma variável de extremo interesse é a sensibilidade dos ativos a mudanças nas taxas de juros ou, mais especificamente, a mudanças na política monetária. Diversos são os canais pelos quais mudanças em taxas de juros são relevantes para os ativos. O canal mais imediato pelo qual aumentos de taxas de juros agem é aumentando o retorno exigido dos ativos, diminuindo o preço dos ativos e levando inicialmente a retornos negativos. Para equities, um canal negativo adicional esperado tipicamente é de queda nos earnings, porque juros maiores têm efeitos marginais negativos para atividade e para a lucratividade das empresas por conseguinte. Para a moeda local, entretanto, espera-se que haja uma apreciação vis-a-vis outras moedas (ou depreciação de outras moedas quando cotadas na moeda doméstica), levando a retornos positivos concomitantes aos aumentos de juros. Será que identificamos estes efeitos para a economia brasileira?

Uma maneira de investigar essa questão é estimar modelos que tenham uma equação do tipo $$R_t^s = \beta^s \Delta i_t + ... + \eta_t^s.$$ O parâmetro de interesse aqui é o $\beta^s$, que para acada ativo $s$ pode ter sinal e magnitude particulares. Para avaliar essa questão, levantei dados diários desde 2003-06 para o IBOV, para o USDBRL e para os fatores de risco de equities disponibilizados pelo [Nefin](https://nefin.com.br/data/risk_factors.html), um centro de pesquisa em Finanças da FEA-USP. Os fatores de risco são excessos de retorno e os retornos diários do IBOV e do USDBRL são subtraídos do CDI do dia. Para as séries de juros, utilizei os 3 primeiros vencimentos de dados de futuros de taxa média de DI de 1 dia. Para calcular os resultados apresentados e discutidos abaixo, utilizei como medida da mudança de juros diária o primeiro componente principal das diferenças diárias das taxas de juros, exceto quando comentado em contrário.[^1]

![Checar se pdf pode ser inserido como imagem. Se não puder, colocar jpeg.](Factor%20Analysis/Figures/ChoquesEstimadosMonePol.jpg)
|                    |   OLS |   EventStudy |   Surpresas |   Rigobon |   Média Surpresas e Rigobon |
|:-------------------|------:|-------------:|------------:|----------:|----------------------------:|
| IBOV menos CDI     | -3.91 |        -5.23 |       -6.13 |     -9.51 |                       -7.82 |
| Retorno de Mercado | -3.45 |        -4.53 |       -5.12 |     -7.16 |                       -6.14 |
| Size               | -0.75 |         0.41 |       -0.29 |     -2.85 |                       -1.57 |
| Value              | -1.03 |        -1.07 |       -1.48 |     -3.46 |                       -2.47 |
| Momentum           | -0.25 |        -0.46 |       -0.74 |      0.59 |                       -0.08 |
| Iliquidez          | -0.07 |         1.11 |        0.75 |     -1.24 |                       -0.24 |
| USDBRL menos CDI   |  1.31 |         1.1  |       -1.19 |     -0.29 |                       -0.74 |

[^1]: Em todos os casos, como é de se esperar e é conhecido para taxas de juros, o primeiro componente principal é muito similar à média simples das séries.

### Estimação por OLS
A princípio, alguém poderia querer estimar a sensibilidade dos ativos a mudanças de juros por meio de um OLS utilizando a amostra inteira, mas isso seria inadequado. O estimador de OLS é viesado se $\Delta i_t$ é endógeno e aqui há um claro caso de endogeneidade por simultaneidade/causalidade reversa: $$\Delta i_t = \alpha^s R_t^s + ... + \varepsilon_t.$$ O ponto em que isso é mais claro é para o câmbio. Tudo mais constante, uma surpresa de depreciação do real deveria aumentar a inflação esperada e, pela expectativa de reação da autoridade monetária, um aumento nas curvas de juros. Não é surpreendente então, que estimemos um efeito de depreciação da moeda quando utilizamos a amostra toda sem endereçar os problemas de endogeneidade. É claramente preciso endereçar este problema.

### Identificação via Event-Study
Uma primeira saída potencial para a identificação seria utilizar a estratégia de event-study ao redor de eventos em que se julgue que a variância dos choques de juros ($\sigma_{\varepsilon,t}^2$) seja muito maior do que a variância dos outros choques estruturais ($\sigma_{\eta^s,t}^2$). Um candidato natural para esses eventos é o dia seguinte às divulgações de decisões do Copom. No Brasil, aproximadamente a cada ~45 dias,[^2] o Copom se reúne para avaliar os rumos da economia, decidir se altera a taxa Selic e expõe algumas opiniões sobre a economia.

Nesta estratégia, temos dois problemas. Tanto pode não ser verdade que a variância dos choques estruturais de juros seja muito maior do que a variância dos retornos, acabando com a estratégia de identificação de um Event-Study, quanto nossa variável é poluída por ser um misto de mudança esperada da taxa de juros (já que os contratos de futuro se referem à taxa média do DI do período e o DI do dia é conhecido antes do comunicado do Copom) e uma legítima surpresa pelo choque de informação que o comunicado e a decisão do Copom podem trazer. Estes dois problemas atrapalham a identificação dos efeitos relevantes e explica bem porque, mesmo com esse novo procedimento, ainda encontramos depreciação do real como consequência de um aumento das taxas curtas de juros.

[^2]: Desde 2006, o Copom faz 8 reuniões por ano. Até 2005, eram feitas 12 reuniões por ano, a cada ~30 dias. Em alguns momentos, houve reuniões extraordinárias.

### Identificação por Surpresas de Juros
Podemos tentar separar o efeito informacional do comunidado das mudanças diárias de juros olhando para as mudanças dos futuros de taxa-média entre o fechamento do mercado anterior ao comunicado do Copom[^3] e a abertura do dia seguinte. Como já se sabia a taxa do DI no horário de fechamento do mercado no dia do comunicado, podemos calcular qual seria a taxa média esperada pelos agentes risco-neutros na abertura do mercado no dia seguinte. Desta forma, assumindo que toda a informação entre o fechamento anterior e a abertura seguinte seja incorporada aos preços diretamente na abertura do mercado, podemos estimar a surpresa que o Copom trouxe naquele dia. Como nos métodos anteriores, estimamos o modelo a partir do primeiro PC das séries de surpresas do Copom nos três primeiros futuros.

Ao contrário do Event-study estimado no passo anterior, aqui temos uma variável plausivelmente exógena, já que o único evento informacional que sistematicamente acontece do fechamento à abertura é o comunicado do Copom. Não é garantido que a surpresa calculada não seja sistematicamente afetada por informações adicionais e que endogeneize nossa medida, mas parece seguro assumir que esse problema seja pequeno e o principal ponto capturado seja a mesmo um choque estrutural na curva de juros. Como esperado para um beta plausivelmente identificado, finalmente estimamos um efeito de apreciação do real como efeito de aumentos da taxa de juros. Além disso, todos os efeitos em equities, quando comparados com os resultados de Event-Study, se tornam mais negativos.

[^3]: Desde 2003-09, o Copom divulga seus comunicados após o fechamento dos mercados, então, ao contrário do que se observa no US, por exemplo, não utilizamos janelas intradiárias para separar surpresa do comunicado.

### Identificação por Heteroscedasticidade
Outra estratégia possível é a identificação por heteroscedasticidade, proposto por Rigobon e Sack (2002) (INSERIR A CITAÇÃO CORRETA AQUI) e que aproveita da maior variância de séries associada aos momentos de divulgação de novos dados econômicos ou financeiros. A ideia é similar à ideia do Event-Study, mas a hipótese de identificação é diferente: em vez de assumir que a variância dos choques estruturais é muito maior *do que a variância dos outros erros*, assume-se que apenas a variância dos choques estruturais de juros aumenta nesses dias, o resto dos choques fica com mesmo variância. A intuição é que, como só a variância dos choques de juros aumentou, poderíamos comparar a covariância das variáveis na amostra dos dias seguintes aos comunicados do Copom com a covariância das variáveis em dias anteriores à divulgação dos dados para estimar os efeitos de juros sobre os retornos diários. Essa comparação é operacionalizada por uma variável instrumental construída aqui com os deltas de juros e também com os retornos dos ativos estudados.

Na nossa especificação, utilizamos os 5 dias anteriores a cada comunicado do Copom para construir a amostra dos dias de controle. Os instrumentos utilizados são construídos como $$w_t^l = \\{ x^l_t * \frac{-1}{T_C - L} | t \in C \\} \cup \\{ x^l_t * \frac{1}{T_T - L} | t \in T \\}, \forall l \in \\{ \Delta i, R_t^1, R_t^2,, ...\\}$$ em que  $T$ é o conjunto de dias seguintes ao Copom, $C$ é o conjunto de pregões anteriores ao comunicado, $T_T$ e $T_C$ são os números de dias em cada conjunto e $L$ é o número de parâmetros estimados no primeiro estágio da estimação (para obtermos um estimador não viesado).
Da mesma forma que na estratégia de identificação por surpresas de juros, obtemos apreciação do real como resultado de aumentos de juros, mesmo que com magnitude menor do que antes. Além disso, todos os efeitos estimados para equities aumentam de magnitude.

### Comentários sobre os Resultados
SESSÃO PRELIMINAR.
Os efeitos acima estimados estão bem alinhados com o esperado.

Primeiro sobre o câmbio. Pensando em termos da UIP, é razoável que um aumento de 1pp no diferencial de juros a favor do Brasil venha associado de apreciação do real em magnitude comparável.

Sobre o efeito em ações, esperamos efeitos fortes em ações para mudanças na curva de juros porque aqui agem dois canais: earnings e fator de desconto. Tanto descontamos os earnings da empresa a uma taxa maior, reduzindo seu valor presente, quanto esperamos redução na atividade, diminuindo os earnings esperados. Não é surpresa, portanto, encontrarmos 5-6% de efeito negativo para o IBOV ou o retorno de mercado em excesso ao CDI.

Para os fatores de risco de equities, entretanto, estamos falando de excessos de retorno comparando equities com equities, então os canais do parágrafo anterior estão presentes nos dois retornos comparados, se anulando. Não é claro porque um rankeamento de size, value, momentum ou iliquidez deveria incorporar mais fortemente os canais de earnings ou fator de desconto no lado long ou no lado short de seus portfolios.

Para o portfolio de Value existe maior discussão sobre sensibilidade a juros porque há "competição" entre os efeitos nas duas pontas do fator de Value(-Growth). Empresas de Value (alto Book/Mkt) seriam tipicamente empresas mais distressed, pras quais o canal de earnings causa mais prejuízo. Empresas de Growth (baixo Book/Mkt) seriam pensadas como mais sensíveis a juros, por terem earnings crescentes no tempo. Essas interpretações são disputáveis (e bastante disputadas).

### Próximos Passos
Rodar com dados semanais em vez de diários
<!--
Teste para instrumentos fracos
 - LEWIS, D. Robust Inference in Models Identified via Heteroskedasticity
-->
External/Internal Instruments / Local Projections para o VAR
 - Mais afeito a modelos low frequency. Semanal já soa overstretch. Cabível,
 é claro, mas overstretch

## Atribuição de Performance
..

# Trading Strategy
..
