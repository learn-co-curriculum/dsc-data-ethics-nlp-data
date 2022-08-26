# Where Does Labeled NLP Data Come From?

* Large language models
* Perpetuating bias/discrimination
* Pattern recognition
* Illegal internet scraping (no consent from org or people)

While web scraping can provide researchers with many data points at once, we must consider how using publicly available data on the internet to train ML comes with consequences we may not expect. 

First, many web scraping activities blur legal boundaries. Some sites clearly state in their terms and conditions that web scraping violates their policies. Moreso, users of a site may post something ‘publicly’, but they may not expect their information to be used in ways data scientists often use it. An instagram post may be public, but many users would feel uncomfortable if their images were collected and used for creating facial recognition software. 

The issue of informed consent is important regardless of the person or organization collecting data. As big data continues to fuel powerful and revenue generating ML systems, we rarely consider that users’ whose data trained these systems are left in the dark and do not profit from the ML built on their information. 

On top of the problems with user consent, models trained on internet data, especially language models are prone to misuse and perpetuating many cognitive biases. For example, GPT-3, a popular language model used to build products like Github Copilot have shown extreme biases when tasked with generating text about Black, muslim, trans, immigrant, and poor people. This happens as the model finds patterns in large datasets of text. Another example of ground truth being unreliable. Though many of the text generations from GPT-3 include demeaning or harmful language, it’s representative of what people say about marginalized groups online. Some of the most popular forums used to train large language models include YouTube comments, 4chan, Reddit, Facebook, and Twitter. 

One of the reasons these language models tend to output such vile text is because of how these social networks moderate and most commonly tolerate hate speech against marginalized people. Often moderators are based in the United States and harbor both western values and cognitive biases. This can lead even moderators, whose job is to assess comments on social media, to rejecting fewer comments made about people who moderators devalue. 

Large language models can be harmful as they perpetuate the beliefs of many internet users, despite attempts by social media companies to curb hate speech. There are many steps that can be taken to avoid perpetuating cognitive biases, this work requires the proper time is spent removing hateful language from datasets. 

Models like GPT-3 and DALLE-E are praised for achieving high accuracy in NLP applications, but these models truly are pattern-matching machines. In addition, it’s very common that the training data is regurgitated and output in language generation tasks. Sometimes this includes copyright, web links, or other obvious signs a chunk of text has been output without topic understanding.

Researchers must use a keen eye to assess if a model truly understands the words it outputs or if it has simply found a pattern match. This process is extremely difficult, even for seasoned NLP practitioners. 


## External Reading

* [The dangers of data scraped from the internet](https://www.technologyreview.com/2021/08/13/1031836/ai-ethics-responsible-data-stewardship/)
* [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)
