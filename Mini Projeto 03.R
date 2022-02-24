#******************************************************************************#
#                                                                              #
#                                                                              #
#                                                                              #
#                                Mini Projeto 03                               #
#                                                                              #
#      Prevendo a Inadimplência de Clientes com Machine Learning e             #
#                                                                              #
#                                                                              #
#******************************************************************************#

# Definindo o Diretório de trabalho
setwd("D:/Dados/Desktop/Data/Curso Data Science Academy/PowerBI/Cap 15")
getwd()

# Definação do problema
# Tá no PDF.

# Instalando os pacotes necessários para o project.
# OBs: precisa ser instalados só uma vez em!!!
install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")

# Carregando os pacotes.
library("Amelia") # Contém funções que tratam valores ausentes.
library("caret") # Pacote ontem contem funções para criações de modelos de Machine Learning
library("ggplot2") # Pacote de criação de gráficos.
library("dplyr") # Pacote para manipular dados.
library("reshape") # Pacote para modificar os formatos dos dados.
library("randomForest") # Pacote usado para Machine Learning.
library("e1071") # Pacote usado para Machine Learning.

# Carregando o dataset
dataset <- read.csv("dataset.csv")

# Visualizando o dataset.
View(dataset)
dim(dataset)
str(dataset)
summary(dataset)

# Removendo a coluna ID do dataset
dataset$ID <- NULL # Apagando a Coluna ID do dataset.
dim(dataset) # Mostra a dimensão do dataset.
View(dataset)

# Renomeando a coluna de classe (Ultima coluna | Váriavel Target)
colnames(dataset) # Mostra o nome de todas as colunas do dataset.
colnames(dataset)[24] <- "inadimplente" # Renomeia a coluna 24.
colnames(dataset) # Confirmando a mudança.
View(dataset) # Visualizando o dataframe.

# Verificando se há valores ausentes.
# Função onde cria um lopping no dataset visualizando se possui valores ausentes..
# no mesmo e mostra soma de todas as contagens que viu em cada coluna.
# Obs. Apenas de valores ausentes.
sapply(dataset, function(x) sum(is.na(x)))
?missmap
# Cria um gráfico onde mostra os valores ausentes dentro do dataset.
missmap(dataset, main = "Valores Missing Observados")
dataset <- na.omit(dataset)

# Convertendo as colunas que estão do tipo int para texto, ou seja, estamos convertendo os atributos
# Genero, escolaridade, estado civil e idade para fatores catégoricos.
# Pois então nesse formato de inteiro, ou seja, o intepretador viu que é uma variavel do tipo númerica
# e presumiu que ela é do tipo quantitativo, mesmo não sendo. Sendo assim isso pode acabar afetando o nosso modelo
# de Machine Learning então temos que mudalas para catégorica.
colnames(dataset)
colnames(dataset)[2] <- "Genero"
colnames(dataset)[3] <- "Escolaridade"
colnames(dataset)[4] <- "Estado_Civil"
colnames(dataset)[5] <- "Idade"
colnames(dataset)
View(dataset)

# Mudando o tipo da coluna Gênero.
str(dataset$Genero)
summary(dataset$Genero)
?cut # Converter variavel número para fator, ou seja, categórica.
# Obs. O valor 0 está sendo usado, para caso exista algum dentro ele vai ser removido!
dataset$Genero <- cut(dataset$Genero, 
                      c(0, 1, 2),
                      labels = c("Masculino", "Feminino"))
View(dataset$Genero)
str(dataset$Genero)

# Mudando o tipo da coluna Escolaridade
str(dataset$Escolaridade)
summary(dataset$Escolaridade)
# Obs. O valor 0 está sendo usado, para caso exista algum dentro ele vai ser removido!
dataset$Escolaridade <- cut(dataset$Escolaridade, 
                      c(0, 1, 2, 3, 4, 5),
                      labels = c("Pós-Graduação", "Graduado", "Ensino Médio", "Outros", "Desconhecido"))
View(dataset$Escolaridade)
str(dataset$Escolaridade)
View(dataset)

# Mudando o tipo da coluna Estado Civil
str(dataset$Estado_Civil)
summary(dataset$Estado_Civil)
dataset$Estado_Civil <- cut(dataset$Estado_Civil, 
                            c(-1, 0, 1, 2, 3),
                            labels = c("Desconhecido", "Casado", "Solteiro", "Outro"))
View(dataset$Estado_Civil)
str(dataset$Estado_Civil)
View(dataset)

# Mudando o tipo da coluna idade
str(dataset$Idade)
summary(dataset$Idade)
hist(dataset$Idade)
dataset$Idade <- cut(dataset$Idade, 
                            c(0, 30, 50, 100),
                            labels = c("Jovem", "Adulto", "Idoso"))
View(dataset$Idade)
str(dataset$Idade)
View(dataset)

# Convertendo a variavel que indica pagamentos para o tipo fator
dataset$PAY_0 <- as.factor(dataset$PAY_0)
dataset$PAY_2<- as.factor(dataset$PAY_2)
dataset$PAY_3 <- as.factor(dataset$PAY_3)
dataset$PAY_4 <- as.factor(dataset$PAY_4)
dataset$PAY_5 <- as.factor(dataset$PAY_5)
dataset$PAY_6 <- as.factor(dataset$PAY_6)

# Dataset após as conversões
str(dataset)
sapply(dataset, function(x) sum(is.na(x))) # Verificando que existem valores nullos
dataset <- na.omit(dataset) # Apagando Valores Nulos
sapply(dataset, function(x) sum(is.na(x))) # COnfirmando que foram apagados.
dim(dataset)

# Alterando a váriavel independente para o tipo fator.
str(dataset$inadimplente)
colnames(dataset)
dataset$inadimplente <- as.factor(dataset$inadimplente)
str(dataset$inadimplente)
View(dataset)

# Total de inadimplentes e não-inadimplentes.
# Mostra a proporção entre os valores que existem dentro de uma coluna no dataset.
table(dataset$inadimplente) 

# Mostrando a proporpoção entre as classes no formato de porcentagem.
prop.table(table(dataset$inadimplente))

# Plot da distribuição usando ggplot2
qplot(inadimplente, data = dataset, geom = "bar") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Set Seed
set.seed(12345)

View(dataset)

# Amostra estratificada
# Seleciona as linhas de acorda com a váriavel inadimplente como strata.
?createDataPartition
indice <- createDataPartition(dataset$inadimplente, p = 0.75, list = FALSE)
dim(indice)

# Definindo os dados de treinamento como subconjunto de dados original
# com números de indice linha (conforme identificado acima) e todas as colunas
dados_treino <- dataset[indice,]
table(dados_treino$inadimplente)

# Vendo porcentagem entre as classes
prop.table(table(dados_treino$inadimplente))

# Núemro de registros do dataset de treinamento
dim(dados_treino)

# Comparando as porcentagens entre as classes de treinamento e dados originais.
compara_dados <- cbind(prop.table(table(dados_treino$inadimplente)),
                       prop.table(table(dataset$inadimplente)))
colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados

# Melt Data - Converte colunas em linhas
?reshape2::melt
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

# Plotando para ver a distribuição de treinamento vs original.
ggplot(melt_compara_dados, aes(x = X1, y = value)) +
  geom_bar( aes(fill = X2), stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Tudo o que não está no dataset de treinamento esta no dataset de teste. Observe o sinal - (menos)
dados_teste <- dataset[-indice,]
dim(dados_teste)
dim(dados_treino)

########################## CONSTRUINDO MODELO DE MACHINE LEARNING ############################
?randomForest
# Do lado esquerdo do ~ é a nossa váriavel Target e a do lado direito é todas as..
# variaveis preditoras
# Variavel Target = Inadimplente.
# Variavel Preditora = Todo o restante das colunas que vão ser usadas para prever.
modelo_v1 <- randomForest(inadimplente ~ ., data = dados_treino)
modelo_v1

# Avaliando modelo
plot(modelo_v1)

# Previsões com dado de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)

# Matriz de confusão
?caret::confusionMatrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$inadimplente, positive = "1")
cm_v1

# Calculando Precision, Recall e F1-Score, métricas de avaliação de modelo preditivo.
y <- dados_teste$inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

# Relação entre precision e recall.
F1 <- (2 * precision * recall) / (precision + recall)
F1 

# Balanceamento de classe
install.packages( "D:/Dados/Downloads/DMwR_0.4.1.tar.gz", repos=NULL, type="source" )
library("DMwR")
# Aplicando o SMOTE - SMOTE: Synthetic Minory Over-sampling Technique
table(dados_teste$inadimplente)
prop.table(table(dados_treino$inadimplente))
set.seed(9560)
dados_treino_bal <- SMOTE(inadimplente ~ ., data = dados_treino)
table(dados_treino_bal$inadimplente)
prop.table(table(dados_treino_bal$inadimplente))

############# - Criando a SEGUNDA versão do modelo. - #################
modelo_v2 <- randomForest(inadimplente ~ ., data = dados_treino_bal)
modelo_v2

# Avaliando modelo
plot(modelo_v2)

# Previsões com dado de teste
previsoes_v2 <- predict(modelo_v2, dados_teste)

# Matriz de confusão
?caret::confusionMatrix
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$inadimplente, positive = "1")
cm_v2

# Calculando Precision, Recall e F1-Score, métricas de avaliação de modelo preditivo.
y <- dados_teste$inadimplente
y_pred_v2 <- previsoes_v2

precision2 <- posPredValue(y_pred_v2, y)
precision2

recall2 <- sensitivity(y_pred_v2, y)
recall2

# Relação entre precision e recall.
F2 <- (2 * precision * recall) / (precision + recall)
F2 

# Importancia das variáveis preditoras para as previsões.
View(dados_treino_bal)
varImpPlot(modelo_v2)

# Importando as váriaveis mais importantes
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var),
                            Importance = round(imp_var[ ,'MeanDecreaseGini'], 2))

# Criando o Rank de variáveis baseado na sua importancia.
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Usando ggplot2 para visualizar a importancia relativa das variáveis.
ggplot(rankImportance,
       aes(x = reorder(Variables, Importance),
           y = Importance,
           fill = Importance)) +
  geom_bar(stat = 'identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust = 0,
            vjust = 0.55,
            size = 4,
            colour = 'red') +
  labs(x = 'Variables') +
  coord_flip()

############# - Criando a TERCEIRA versão do modelo. - #################
colnames(dados_treino_bal)
modelo_v3 <- randomForest(inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1,
                          data = dados_treino_bal)
modelo_v3

# Avaliando modelo
plot(modelo_v3)

# Previsões com dado de teste
previsoes_v3 <- predict(modelo_v3, dados_teste)

# Matriz de confusão
?caret::confusionMatrix
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$inadimplente, positive = "1")
cm_v3

# Calculando Precision, Recall e F1-Score, métricas de avaliação de modelo preditivo.
y <- dados_teste$inadimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

# Relação entre precision e recall.
F1 <- (2 * precision * recall) / (precision + recall)
F1

# Salvando o modelo em Disco.
# Obs: está salvo dentro da pasta modelo.
saveRDS(modelo_v3, file = "modelo/modelo_v3.rds")

# Carregando o modelo.
# Obs: que está salvo no nosso pc.
modelo_final <- readRDS("modelo/modelo_v3.rds")
  
# Fazendo previsões com novos dados = 3 novos clientes.

# Dados dos clientes para realizar as previsões, sem a váriavel inadimplente.
PAY_0 <- c(0, 0, 0)
PAY_2 <- c(0, 0, 0)
PAY_3 <- c(1, 0, 0)
PAY_AMT1 <- c(1100, 1000, 1200)
PAY_AMT2 <- c(1500, 1300, 1150)
PAY_5 <- c(0, 0, 0)
BILL_AMT1 <- c(350, 420, 280)

# Concatenando os dados em um dataframe.
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)

# Previsões do modelo
# Obs: O erro que foi imprimido na tela foi causado pelo motivo de que..
# os tipos de váriaveis do dataset de treino e dos novos clientes..
# são diferentes, então é necessário fazer uma conversão nos novos dados...
# para assim poder rodar o modelo.
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)

# Conferindo os tipos de dados dos dois dataset.
str(dados_treino_bal)
str(novos_clientes)

# Conrventendo os tipos de dados dos clientes novos.
# Obs: Apenas convertendo de maneira convencional, não irá funcionar..
# pois no nosso dataset de treino, as váriaveis tem diferentes níveis dentro dela..
# então para funcionar de maneira corretamente temos que trazer todos esses niveis para..
# os novos dados que vão ser convertido, e é isso que está sendo feito abaixo:
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_bal$PAY_5))
str(novos_clientes)

# Depois te realizar todas as mudanças, agora sim poderá ser feita as previsões.
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
View(previsoes_novos_clientes)


