### Tabela de Conteúdos
1. [Introdução](#introdução)
2. [Yield Curve](#yield-curve)
    1. [Nelson-Siegel-Svensson](#nelson-siegel-svensson)
    2. [Prêmio de Risco da Curva](#prêmio-de-risco-da-curva)
        1. [Histórico de Prêmios de Risco](#histórico-de-prêmios-de-risco)
        2. [Métricas de Replicação](#métricas-de-replicação)
4. [Factor Analysis](#factor-analysis)
    1. [Efeitos de Política Monetária](#efeitos-de-política-monetária)
        1. [Estimação por OLS](#estimação-por-ols)
        2. [Identificação via Event-Study](#identificação-via-event-study)
        3. [Identificação por Surpresas de Juros](#identificação-por-surpresas-de-juros)
        4. [Identificação por Heteroscedasticidade](#identificação-por-heteroscedasticidade)
        5. [Comentários sobre os Resultados](#comentários-sobre-os-resultados)
   <!--
        6. [Próximos Passos](#próximos-passos)
   -->
    2. [Asset Pricing Trees](#asset-pricing-trees)
<!--
    3. [Fronteira Eficiente](#fronteira-eficiente)
    4. [Atribuição de Performance](#atribuição-de-performance)

4. [Trading Strategy](#trading-strategy)
-->
# Introdução
Neste repositório eu concentro alguns estudos realizados (ou suas etapas preliminares).

A parte mais avançada hoje é a modelagem de curva de juros, [Yield Curve](#yield-curve). A partir das classes _YieldCurve_, que concentra os métodos básicos para transformar e visualizar curvas de juros, e _AffineYieldCurveModel_, que concentra os métodos básicos para modelos com SDF e curva lineares nos mesmos estados, implemento o modelo de [Adrian, Crump and Moench (2013)](https://doi.org/10.1016/j.jfineco.2013.04.009) para decompor as taxas de juros nominal observadas na curva US (ou de qualquer outro país de interesse) entre a expectativa risco-neutra da rolagem das taxas curtas e o prêmio de risco cobrado para cada horizonte.[^projAdicionalCurva]

[^projAdicionalCurva]: Uma extensão que estou desenvolvendo à frente é a de decompor conjuntamente a curva de juros nominais e a curva de juros reais, permitindo criar também uma decomposição da inflação implícita (breakeven).

Na seção seguinte, [Factor Analysis](#factor-analysis), há alguns projetos que utilizam fatores de risco do [Nefin](https://nefin.com.br/data/risk_factors.html) tanto para estimar os prêmios de risco de outros ativos, quanto como portifólios de equities nos quais buscamos identificar efeitos de política monetária por diferentes métodos de identificação, que consigam separar os efeitos causais de mudanças nas taxas de juros nos retornos dos ativos.[^projAdicionalFatores]

[^projAdicionalFatores]: Um projeto adicional a ser desenvolvido é o de [atribuição de performance a índices setoriais e fundos](#atribuição-de-performance) (cálculo de alphas e betas). Para este projeto, falta basicamente levantar um histórico de performance de fundos de investimento, mas ainda não fui atrás de uma fonte adequada para lidar com viés de sobrevivência. 

O projeto ora em desenvolvimento é um subprojeto de Factor Analysis, que replica o método de [Bryzgalova, Pelger e Zhu (2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3493458) para estimar os portfolios teste ótimos para um modelo de fatores. Os autores argumentam que os portfolios assim obtidos são (1) estimados rapidamente (o que se constata na velocidade de estimação das árvores individuais) e (2) apresentam um barra mais alta para a validação de potenciais fatores, já que conseguiriam _span_ um espaço de retornos mais complexo do que portfolios de double- ou triple-sorting. A ideia é construir os portfolios-teste por um método data-drive, que escolhe escolhe os melhores portfolios long-only pela maximização do SR implícito com robustez dupla: com shrinkage da média de retornos dos portfolios-teste e da matriz de covariância entre eles. Para tanto, os portfolios são criados via árvore de decisões e splits por características dos ativos, generalizando os processos de sorting por características. Partindo do portfolio de mercado, dividem-se os ativos em dois portfolios separados pela mediana de uma característica e estima-se o SR ótimo. Se o split for bem-sucedido (aumentar o SR na amostra de validação), então se repete o processo, aumentando a árvore e o número de portfolios de teste até que não haja mais ramificações proveitosas adicionais.

# Yield Curve
## Nelson-Siegel-Svensson
O workhorse básico para as manipulações de curvas de juros, além da classe _YieldCurve_, que guarda os métodos genéricos relativos a curvas de juros (visualização, conversões entre taxas, log-taxas, log-pu, holding period return, excess return, ...), é o modelo Nelson-Siegel-Svensson[^NSS]. Este é um modelo flexível que permite construir curvas de juros a partir de 4 a 6 parâmetros para qualquer vértice que se queira e a flexibilidade vem das duas "concavidades" que o modelo completo se permite ter: $$i_t(n) = \beta_{0t} + \beta_{1t} \frac{1 - exp(-n/\lambda_{1t})}{n/\lambda_{1t}} + \beta_{2t} \left(\frac{1 - exp(-n/\lambda_{1t})}{n/\lambda_{1t}} - exp(-n/\lambda_{1t}) \right) + \beta_{3t} \left(\frac{1 - exp(-n/\lambda_{2t})}{n/\lambda_{2t}} - exp(-n/\lambda_{2t}) \right).$$ Além disso, optou-se por usar esta implementação básica para construção de curvas de juros pela disponibilidade de curvas de juros utilizando este modelo. Além das curvas da Anbima utilizarem esse modelo, o Fed também disponibiliza versões da curva estimadas para a [curva de juros nominal](#https://www.federalreserve.gov/data/nominal-yield-curve.htm) e a [curva de juros reais](#https://www.federalreserve.gov/data/tips-yield-curve-and-inflation-compensation.htm) para US^[. Estes dados são utilizados para construir os exercícios de estimação do modelo de [Adrian, Crump e Moench (2013)](#https://www.sciencedirect.com/science/article/abs/pii/S0304405X13001335) para a estimação dos Bond-Risk Premium e Expectativa Risco-Neutra embutidos na curva de juros nominal US e também o modelo complementar a este, o [Abrahams, Adrian, Crump e Moench (2016)](#https://doi.org/10.1016/j.jmoneco.2016.10.006), que analisa conjuntamente a curva nominal e a curva real, com implicações de previsão para a breakeven.

A implementação do Nelson-Siegel-Svensson está na classe _GSK\_Curve_, por conta de [Gürkaynak, Sack e Wright (2008)](#https://www.federalreserve.gov/pubs/feds/2008/200805/200805abs.html), que estimaram este modelo para US e começaram o projeto para disponibilização desses dados pelo Fed.

![Yields - US](Yield%20Curve/Figures/Yields.jpg)
![FRAs de 1 ano - US](Yield%20Curve/Figures/FRAs.jpg)

[^NSS]: Implementação baseada no [Working Paper do NBER ](#https://www.nber.org/papers/w4871), com versão publicada: "Estimating Forward Interest Rates with the Extended Nelson and Siegel Method," Sveriges Riksbank Quarterly Review 1995:3, pp 13-26.

## Prêmio de Risco da Curva
[Adrian, Crump e Moench (2013)](#https://www.sciencedirect.com/science/article/abs/pii/S0304405X13001335) utilizam um modelo afim da estrutura termo da curva de juros e constróem uma série de argumentos para (1) utilizar os dados observados da curva de juros para caracterizar os estados relevantes para os retornos obtidos de títulos em cada vértice da curva, (2) para utilizar 3 regressões lineares para estimar os prêmios de risco ($\lambda_t = \lambda_0 + \lambda_1 X_t$) associado a cada estado e (3) construir testes de hipótese para entender quais estados são relevantes para prêmio de risco significante. Além disso, pela natureza do modelo deles, títulos em cada vértice têm log-preço (e portanto log-taxas) afins nos estados da economia $$\ln P_t(n) = A_t(n; \lambda_t) + B_t(n;\lambda_t) X_t, $$ então, impondo $\lambda_{1} = 0 \implies \lambda_t = \lambda_0$, é possível estimar os preços que refletem apenas a rolagem esperada da taxa de juros $$\ln P_t^{RF}(n) = A_t^{RF}(n; \lambda_0) + B_t^{RF}(n;\lambda_0) X_t .$$ Convertendo esses log-preços a log-taxas e calculando sua diferença, podemos calcular o Bond Risk Premium (BRP).

![Decomposição da Curva](Yield%20Curve/Figures/Decomposicao%20da%20Curva.jpg)

### Histórico de Prêmios de Risco
Além de calcular a decomposição da última curva disponível, podemos avaliar o comportamento histórico dos prêmios de risco da curva.

O primeiro e mais saliente feature é o quase perfeito rankeamento entre os prêmios de risco pela horizonte de maturidade do investimento. Isso é intuitivo: quanto maior o horizonte de investimento, menor é a convicção de que as taxas de juros seguirão o caminho esperado e provavelmente maiores são os prêmios cobrados para carregar esse risco. A ordem entre esses prêmios se altera em alguns momentos, mas o padrão predominante é o ordenamento.  

Outro ponto interessante é a queda histórica dos BRPs estimados em todas as maturidades selecionadas, mesmo que não uniformemente no tempo. Parte dessa diminuição é mecanicamente explicada pela queda das taxas de juros e a expectativa de que as taxas de juros se manteriam baixas, no zero lower bound (ZLB), por muito tempo. Mais ainda, em alguns momentos mais conturbados como na crise da Covid-19, o prêmio de risco estimado é negativo, condizente com os agentes quererem pagar para tomar o risco de rolagem, mas eliminar o risco associado a eventos extremos. Aos poucos, parece que estamos voltando ao mundo de prêmios de risco positivos.

![BRP - Anos Selecionados](Yield%20Curve/Figures/BRP%20-%20Anos%20Selecionados.jpg)

### Métricas de Replicação
Infelizmente, os dados fornecidos pelo Fed para a curva de juros sofre revisões potenciais a cada divulgação, então, após 10 anos de novos dados e reestimações, os dados do recorte dos autores (1987M01 a 2011M12) não estão disponíveis livremente. A implementação do modelo realizada, entretanto, se encaixa bastante bem no padrão obtido pelos autores originais. Aplicando o modelo aos dados de 1987 a 2011, percebemos na tabela abaixo o mesmo padrão que os autores destacaram na Tabela 2 de seu paper sobre os erros do modelo nas maturidades selecionadas:
1. Erros de precificação dos yields: erros pequenos, mas altamente correlacionados serialmente.
2. Erros de precificação dos retornos: erros pequenos, com variados níveis de assimetria e curtose, mas principalmente não correlacionados serialmente.

A interpretação aqui é a padrão de séries temporais: porque os resíduos das regressões dos retornos não são correlacionados serialmente, conseguimos capturar o que existe de previsibilidade neles. Caso mais variáveis sejam incluídas, podemos melhorar a projeção dos retornos futuros, por exemplo, incluindo features não lineares dos dados, mas o modelo é bem sucedido em capturar o prêmio de risco que o modelo parcimonioso se propõe a capturar.

|                                | 12     | 24     | 36     | 60     | 84     | 120    |
|:-------------------------------|:-------|:-------|:-------|:-------|:-------|:-------|
| Panel A: Yield pricing errors  |        |        |        |        |        |        |
| Mean                           | -0.0   | -0.001 | -0.001 | -0.002 | -0.003 | -0.004 |
| Std. Deviation                 | 0.006  | 0.006  | 0.006  | 0.006  | 0.005  | 0.008  |
| Skewness                       | -0.356 | 0.383  | 0.057  | -0.15  | 0.147  | -0.109 |
| Kurtosis                       | 0.226  | 0.01   | -0.317 | -0.546 | -0.662 | -0.533 |
| $\rho(1)$                      | 0.764  | 0.835  | 0.912  | 0.941  | 0.938  | 0.856  |
| $\rho(6)$                      | 0.533  | 0.634  | 0.791  | 0.78   | 0.832  | 0.513  |
| Panel B: Return pricing errors |        |        |        |        |        |        |
| Mean                           | -0.0   | -0.0   | -0.0   | -0.0   | -0.0   | -0.0   |
| Std. Deviation                 | 0.0    | 0.0    | 0.0    | 0.0    | 0.0    | 0.0    |
| Skewness                       | 3.562  | 1.9    | 4.26   | 2.034  | 0.278  | 0.339  |
| Kurtosis                       | 37.417 | 15.155 | 48.229 | 18.175 | 2.432  | 3.337  |
| $\rho(1)$                      | 0.038  | 0.313  | 0.119  | -0.017 | -0.01  | -0.119 |
| $\rho(6)$                      | 0.119  | 0.158  | 0.159  | 0.048  | 0.051  | 0.006  |

![ACM (2013) - Table 2](Yield%20Curve/Figures/ACM_2013_Table_02.png)




# Factor Analysis

## Efeitos de Política Monetária
Uma variável de extremo interesse é a sensibilidade dos ativos a mudanças nas taxas de juros ou, mais especificamente, a mudanças na política monetária. Diversos são os canais pelos quais mudanças em taxas de juros são relevantes para os ativos. O canal mais imediato pelo qual aumentos de taxas de juros agem é aumentando o retorno exigido dos ativos, diminuindo o preço dos ativos e levando inicialmente a retornos negativos. Para equities, um canal negativo adicional esperado tipicamente é de queda nos earnings, porque juros maiores têm efeitos marginais negativos para atividade e para a lucratividade das empresas por conseguinte. Para a moeda local, entretanto, espera-se que haja uma apreciação vis-a-vis outras moedas (ou depreciação de outras moedas quando cotadas na moeda doméstica), levando a retornos positivos concomitantes aos aumentos de juros. Será que identificamos estes efeitos para a economia brasileira?

Uma maneira de investigar essa questão é estimar modelos que tenham uma equação do tipo $$R_t^s = \beta^s \Delta i_t + ... + \eta_t^s.$$ O parâmetro de interesse aqui é o $\beta^s$, que para acada ativo $s$ pode ter sinal e magnitude particulares. Para avaliar essa questão, levantei dados diários desde 2003-06 para o IBOV, para o USDBRL e para os fatores de risco de equities disponibilizados pelo [Nefin](https://nefin.com.br/data/risk_factors.html), um centro de pesquisa em Finanças da FEA-USP. Os fatores de risco são excessos de retorno e os retornos diários do IBOV e do USDBRL são subtraídos do CDI do dia. Para as séries de juros, utilizei os 3 primeiros vencimentos de dados de futuros de taxa média de DI de 1 dia. Para calcular os resultados apresentados e discutidos abaixo, utilizei como medida da mudança de juros diária o primeiro componente principal das diferenças diárias das taxas de juros, exceto quando comentado em contrário.[^1]

![Estimativas dos Choques Estimados](Factor%20Analysis/Figures/ChoquesEstimadosMonePol.jpg)
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
Os efeitos acima estimados estão bem alinhados com o esperado quando nos atentamos aos problemas de identificação possíveis em um OLS simples.

Primeiro sobre o câmbio. Pensando em termos da UIP, é razoável que um aumento de 1pp no diferencial de juros a favor do Brasil venha associado de apreciação do real em magnitude comparável. O fato de obtermos resultados contrários ao esperado quando não utilizamos boas técnicas de identificação dos efeitos causais são testemunhas claras da necessidade de buscar uma fonte de variação exógena para as estimações.

Sobre o efeito em ações, esperamos efeitos fortes em ações para mudanças na curva de juros porque aqui agem dois canais: earnings e fator de desconto. Tanto descontamos os earnings da empresa a uma taxa maior, reduzindo seu valor presente, quanto esperamos redução na atividade, diminuindo os earnings esperados. Não é surpresa, portanto, encontrarmos 5-6% de efeito negativo para o IBOV ou o retorno de mercado em excesso ao CDI.

Para os fatores de risco de equities, entretanto, estamos falando de excessos de retorno comparando equities com equities, então os canais do parágrafo anterior estão presentes nos dois retornos comparados, se anulando. Não é claro porque um rankeamento de size, momentum ou iliquidez deveria incorporar mais fortemente os canais de earnings ou fator de desconto no lado long ou no lado short de seus portfolios.

Para o portfolio de Value existe maior discussão sobre sensibilidade a juros porque há "competição" entre os efeitos nas duas pontas do fator de Value(-Growth). Empresas de Value (alto Book/Mkt) seriam tipicamente empresas mais distressed, pras quais o canal de earnings causa mais prejuízo. Empresas de Growth (baixo Book/Mkt) seriam pensadas como mais sensíveis a juros, por terem earnings crescentes no tempo. Importa salientar que essas interpretações são disputáveis (e bastante disputadas).

<!--
### Próximos Passos
Rodar com dados semanais em vez de diários

Teste para instrumentos fracos
 - LEWIS, D. Robust Inference in Models Identified via Heteroskedasticity

External/Internal Instruments / Local Projections para o VAR
 - Mais afeito a modelos low frequency. Semanal já soa overstretch. Cabível,
 é claro, mas overstretch
-->

## Asset Pricing Trees
Aqui replicamos o método do working paper ["Forest Through the Trees: Building Cross-Sections of Stock Returns" de Bryzgalova, Pelger e Zhu](#https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3493458). Os autores têm uma crítica bastante pertinente à literatura de fatores em ações: a construção dos test-assets, com cujos retornos avaliamos os candidatos a fatores, é arbitrária. Por que construir portfolios por double- ou triple-sorting de características específicas seria a solução? Se os test-assets forem mal construídos, no sentido de não conseguirem projetar as dimensões relevantes do fator de desconto estocástico, então a barra para um candidato a fator conseguir explicar o SDF possível com os test assets é baixa. Minha intuição é que o SDF é um objeto complexo e que, para atingir o Sharpe-Ratio esperado máximo dado pelo SR esperado do SDF, é preciso buscar um conjunto de test assets que consiga capturar padrões eventualmente infrequentes/raros e que apenas um subconjunto de ativos específicos consegue capturar. A solução então é buscar um método data-driven que ao mesmo tempo generaliza a ideia de double- e triple-sorting, já que isso torna o conjunto de test assets mais interpretáveis, mas que seja orientado para efetivamente capturar a correta precificação de padrões complexos de retornos.

O método e a contribuição dos autores se baseia em dois pilares: estimações robustas das covariâncias e retornos esperados por meio de shrinkages diversos e _pruning_ da árvore de decisão baseado no Sharpe-Ratio obtido pela árvore completa de portfolios, em oposição às técnicas habituais de estimação de árvores de decisão, que usariam critérios locais para split de portfolios. Partindo do conjunto de retornos dos ativos disponíveis e os market caps e características de cada ativo (como size, book-to-market, ...), os autores propõem criar portfolios value-weighted a partir de splits sequenciais, começando pelo portfolio de mercado, e sempre splittando os portfolio pela mediana de alguma das características disponíveis. Além dos parâmetros de shrinkage sobre os quais os autores otimizam, a ordem das características disponíveis é extremamente relevante e define a árvore ótima obtida. Por conta disso, os autores estimam as árvores ótimas para cada possível ordem de características (utilizando amostras de treino e validação), e então comparam as diferentes árvores de portfolios ótimas obtidas no processo pelas métricas na amostra de teste.

<!--
## Atribuição de Performance
..
-->
<!--
4. [Trading Strategy](#trading-strategy)
# Trading Strategy
..
-->

