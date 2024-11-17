library(ggplot2)

# sort by count
amazon_review_ID.shuf.lrn$Class <- reorder(amazon_review_ID.shuf.lrn$Class,amazon_review_ID.shuf.lrn$Class,FUN=length)

# plot
ggplot(amazon_review_ID.shuf.lrn, aes(x=Class)) +
  geom_histogram(fill = "steelblue4", color = "lightgray", stat = "count") + 
  coord_flip() +
  theme(axis.text.y=element_text(size=rel(0.8)),
        axis.text.x=element_text(size=rel(0.8)), 
        axis.title.y=element_blank(),
        axis.title.x=element_blank())
  