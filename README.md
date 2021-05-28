# crossner

<img src="http://latex.codecogs.com/gif.latex?\begin{table}[!htbp]
	\centering\resizebox{\linewidth}{!}
	{
		\begin{tabular}{|c|ccc|ccc|ccc|}
			\hline
			&& NCBI-Disease& &&BC5CDR-Disease& &&2010 i2b2/VA&  \\
			\hline
			Model & Precision & Recall & F1-score & Precision & Recall & F1-score& Precision & Recall & F1-score\\
			\hline
			BERT(2018) \cite{devlin2018bert}& 83.44&	87.18&	85.27&82.40	&84.17&	83.28 &84.04&84.08&84.06 \\
			
			RoBERTa(2019) \cite{liu2019roberta}&84.65&	85.62&	85.13& 79.17&	81.89&	80.51& --&--&--\\
			
			BioBERT(2020) \cite{lee2020biobert}& 88.15&	91.36&	89.71&84.84	&\textbf{87.95}&86.37& 85.37&85.64&85.51\\
			Spark NLP(2021) \cite{kocaman2020biomedical}& --&	--&	89.13&--&--&	--& --&--&--\\
			RDANER(2020) \cite{houjin2020rdaner} &--&--&87.39 &--&--&--&--&--&-- \\
			ELMO(2019) \cite{el2019embedding}&--&--&--&--&--&--&--&--&86.23\\
			TENER-Bio&  88.09& 90.93 &89.49&84.65&87.05&85.83&  80.44&82.04&81.27\\
			\hline
			TENER-Bio-CW& \textbf{88.99} &\textbf{91.77}  &\textbf{90.36}&\textbf{85.89}&87.20&\textbf{86.57}&\textbf{88.17}&\textbf{87.86}&\textbf{88.01}  \\
			\hline
		\end{tabular}
	}
	\\
	\caption{Performance comparison with the SOTA models.}\label{sota}
\end{table}" />
