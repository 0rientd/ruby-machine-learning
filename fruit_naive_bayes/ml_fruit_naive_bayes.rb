require 'rumale'

# Cada item representa uma fruta com as seguintes características: [arredondada?, suculenta?, vermelha?, doce?, nome] (valores binários)
data = [
          [0,1,1,1,'morango'],
          [1,0,0,0,'limao'],
          [1,1,0,1,'pera'],
          [0,0,0,1,'banana'],
          [1,1,1,1,'cereja'],
          [1,1,1,0,'tomate'],
          [1,1,0,1,'maca']
        ]

def treinar_modelo_de_frutas(data)
  # Separando as características (features) da fruta e o rótulo (label)
  x = data.map { |x| x[0..3].map(&:to_f) }
  y = data.map { |x| x[4] }

  # É necessário codificar os rótulos (nomes das frutas) em valores numéricos
  @encoder = Rumale::Preprocessing::LabelEncoder.new
  y_encoded = @encoder.fit_transform(y)

  # Convertendo as características para uma matriz Numo::DFloat, como exigido pelo Rumale
  x_matrix = Numo::DFloat[*x]

  # Treinando o classificador Naive Bayes com os dados
  @estimator = Rumale::NaiveBayes::ComplementNB.new(smoothing_param: 1.0)
  @estimator.fit(x_matrix, y_encoded)
end

# Exemplo de teste
# Criando um exemplo com características específicas e transformando-o para o formato aceito pelo Rumale
sample = Numo::DFloat[[1, 1, 0, 1]]

# Chamando método para saber a previsão da fruta escolhida com base nas caracteristicas
pred = @estimator.predict(sample)

# Usando o codificador para converter a previsão numérica de volta ao nome da fruta
label = @encoder.inverse_transform(pred)

# Código adaptado do curso de ML do https://github.com/TeoMeWhy
# Link do curso: https://youtube.com/playlist?list=PLvlkVRRKOYFR6_LmNcJliicNan2TYeFO2&si=W0-U12cYOJdvQoXo