---
title: Crawling the Web with LLM
authors:
  - BZ
date: 2025-02-16
categories: 
  - LLM
---

## Crawling the Web with Large Language Models (LLMs)

<!-- more -->

**Frameworks**

- Skyvern
- *ScrapegraphAI*
- Crawl4AI
- Reader
- Firecrawl
- Markdowner

**Code Examples**

```python linenums="1"
import json

from scrapegraphai.graphs import SmartScraperGraph

graph_config = {
    "llm": {
        "model": "ollama/phi4",
        "temperature": 0,
        "format": "json",  # Ollama needs the format to be specified explicitly
        "base_url": "http://localhost:11434",  # set Ollama URL
    },
    "embeddings": {
        "model": "ollama/jina/jina-embeddings-v2-base-en",
        "base_url": "http://localhost:11434",  # set Ollama URL
    }
}

smart_scraper_graph = SmartScraperGraph(
    prompt="Extract useful information from the webpage, including a description of what the company does, founders and social media links",
    source="https://scrapegraphai.com/",
    config=graph_config
)

result = smart_scraper_graph.run()
print(json.dumps(result, indent=4))

```

**Output**:
```json
{
    "title": "ScrapeGraphAI: AI-Powered Web Scraping Made Easy",
    "description": "ScrapeGraphAI is a cutting-edge tool designed to simplify web scraping by leveraging Large Language Models (LLMs). It offers an easy-to-use API that allows developers and AI agents to extract structured data from various types of websites without the need for complex coding. With features like handling website changes, scalability, and integration with popular programming languages such as Python, JavaScript, and TypeScript, ScrapeGraphAI is perfect for anyone looking to harness web data efficiently.",
    "features": {
        "ease_of_use": "ScrapeGraphAI provides a user-friendly API that requires minimal coding effort, making it accessible even for those without extensive technical expertise.",
        "integration": "The tool seamlessly integrates with Python, JavaScript, and TypeScript, allowing developers to incorporate web scraping into their existing projects effortlessly.",
        "website_handling": "Capable of handling diverse websites including e-commerce platforms, social media sites, blogs, forums, and more. It efficiently manages dynamic content loaded via JavaScript or AJAX.",
        "adaptability": "ScrapeGraphAI is designed to adapt to website changes automatically, reducing the need for manual maintenance and ensuring consistent data extraction.",
        "performance_and_reliability": "The platform offers high performance and reliability, with features like rate limiting, IP rotation, and CAPTCHA solving to ensure smooth operation even under challenging conditions."
    },
    "pricing": {
        "free_plan": "Includes 1000 requests per month, suitable for small-scale projects or testing purposes.",
        "paid_plans": [
            {
                "starter": "$9.99/month",
                "requests": "10,000"
            },
            {
                "pro": "$49.99/month",
                "requests": "50,000"
            },
            {
                "enterprise": "Custom pricing",
                "requests": "Unlimited"
            }
        ]
    },
    "founders": {
        "marco_perini": "Founder & CTO - [LinkedIn](https://www.linkedin.com/in/perinim/)",
        "marco_vinciguerra": "Founder & Software Engineer - [LinkedIn](https://www.linkedin.com/in/marco-vinciguerra-7ba365242/)",
        "lorenzo_padoan": "Founder & CEO - [LinkedIn](https://www.linkedin.com/in/lorenzo-padoan-4521a2154/)"
    },
    "contact_info": {
        "email": "contact@scrapegraphai.com",
        "social_media": {
            "linkedin": "[ScrapeGraphAI LinkedIn](https://www.linkedin.com/company/101881123)",
            "twitter": "[Twitter](https://x.com/scrapegraphai)",
            "github": "[GitHub](https://github.com/ScrapeGraphAI/Scrapegraph-ai)",
            "discord": "[Discord](https://discord.gg/uJN7TYcpNa)"
        }
    },
    "legal_info": {
        "privacy_policy": "[Privacy Policy](/privacy)",
        "terms_of_service": "[Terms of Service](/terms)",
        "api_status": "[API Status](https://scrapegraphapi.openstatus.dev)"
    },
    "footer": "\u00a9 2025 - ScrapeGraphAI, Inc"
}
```


