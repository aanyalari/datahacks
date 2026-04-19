#California Ocean Health Monitor

Ocean acidification threatens California's $3B fishing industry, but resource managers lack tools that translate raw sensor data into actionable, location-specific warnings. We built a real-time ocean health dashboard pulling from two data sources: the Scripps Institution of Oceanography CCE LTER Mooring Network (pH, temperature, dissolved oxygen, chlorophyll, nitrate at stations CCE1 and CCE2) and the CalCOFI fish larvae survey dataset spanning 1951–2023.

The platform answers three questions fisheries managers cannot currently answer: where is acidification worst, when will biological thresholds be crossed, and is the ecosystem damage keeping pace with the chemistry?

Our ML pipeline combines HistGradientBoosting for 6–72 hour pH forecasting, ARIMA for seasonal trend decomposition, and Isolation Forest for anomaly detection flagging acute stress events in real time. Key finding: CCE2 has recorded 250 pH breach events below the anchovy larvae survival threshold of 7.9 since 2011, versus 4 events at offshore CCE1 — a 60x differential that points to disproportionate nearshore risk.

Deployed on AWS with a PostgreSQL backend and Streamlit frontend.