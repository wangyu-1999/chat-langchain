---
title: News Clustering Analysis
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 新闻聚类分析系统

这是一个基于 FastAPI 和 Docker 的新闻聚类分析系统，可以自动获取、处理和分析新闻数据。

## 环境变量

需要在 Hugging Face Space 的 Secrets 中设置以下环境变量：

- AZURE_STORAGE_CONNECTION_STRING
- ZILLIZ_CLOUD_URI
- ZILLIZ_CLOUD_TOKEN
- ZHIPU_API_KEY
- API_KEY

## 功能

- 新闻数据摄入
- 新闻聚类分析
- REST API 接口

## API 文档

部署后可访问 `/docs` 路径查看完整的 API 文档。

## 技术栈

- FastAPI
- Docker
- Azure Table Storage
-