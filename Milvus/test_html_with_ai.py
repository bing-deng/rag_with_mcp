   from html_to_milvus import HTMLToMilvusProcessor
   
   processor = HTMLToMilvusProcessor()
   processor.connect_to_milvus()
   processor.create_collection()
   
   # 处理您的 HTML 内容
   content_blocks = processor.parse_html(your_html_content, source_url)
   processor.insert_html_content(content_blocks)
   
   # 搜索
   results = processor.search_content("您的搜索查询")