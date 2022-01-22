RUL_Pred <- read.csv("TransformedData.csv")
RUL_Pred$Source <- as.factor(RUL_Pred$Source)
anova <- lm(RUL ~ Source, data = RUL_Pred)
plot(anova, which=1)
