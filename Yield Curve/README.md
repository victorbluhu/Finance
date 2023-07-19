Este repositório reúne scripts para modelagem de curvas de juros e modelagem delas.

O arquivo YieldCurveLib é a lib de funções e classes para:
* baixar os arquivos relevantes do site do Fed (histórico da curva desde 1961),
* recuperar e montar o dataset de parâmetros
* definir o objeto YieldCurve, que reúne os métodos relevantes para trabalhar com curvas de juros, incluindo visualizações relevantes e transformações, por exemplo, entre os objetos abaixo:
    - yields
    - logyields
    - discount factor
    - log discount factor
    - holding period returns
    - excess return
* montar uma YieldCurve com base nos parâmetros do modelo de Nelson-Siegel-Svenson
* estimar o modelo de Adrian, Crump and Moench (2013) para curvas de juros.

Incluir a citação correta do paper "Pricing the Term Structure with Linear Regressions".
