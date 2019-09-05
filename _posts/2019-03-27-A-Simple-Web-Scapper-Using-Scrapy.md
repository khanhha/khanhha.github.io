---
title: "A Simple Web Scraper Using Scrapy"
categories:
  - posts
tags:
  - web_scrape
---

This tutorial is about the general pipeline of a simple web scrapper based on Scrapy, a popular python scraping library. The main focus will be about two main mechanisms supported by Scrapy to extract information from the HTML data.

# General Pipeline.
First, I will quickly explain the basic flow of a Spider class that extract quote texts from two URLs, as shown in the below code. For visualization, a simple version of the downloaded HTML files from these two URLs will look as below.

```html
# HTML file is taken from https://docs.scrapy.org
<div class="quote">
    <span class="text">“The world as we have created it is a process of our
    thinking. It cannot be changed without changing our thinking.”</span>
    <span>
        by <small class="author">Albert Einstein</small>
        <a href="/author/Albert-Einstein">(about)</a>
    </span>
    <div class="tags">
        Tags:
        <a class="tag" href="/tag/change/page/1/">change</a>
        <a class="tag" href="/tag/deep-thoughts/page/1/">deep-thoughts</a>
        <a class="tag" href="/tag/thinking/page/1/">thinking</a>
        <a class="tag" href="/tag/world/page/1/">world</a>
    </div>
</div>
```

To extract information from this HTML file, we need to write a Spider class as shown the below example. The built-in variable __start_urls__ in this class defines a set of URLs from which data will be crawled. Scrapy will automatically submit HTTP requests to these URLs, and when the response is available, calls the function __parse__.

The main tasks now will be how to extract the information we need from this response, which is done by the function __parse__ using CSS selector, the topic of the next paragraph.

```python

#this code is taken from https://docs.scrapy.org

import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = [
        'http://quotes.toscrape.com/page/1/',
        'http://quotes.toscrape.com/page/2/',
    ]
    def parse(self, response):
        for quote in response.css('div.quote'):
            yield {
                'text': quote.css('span.text::text').get(),
                'author': quote.css('small.author::text').get(),
                'tags': quote.css('div.tags a.tag::text').getall(),
            }
```


# CSS selector
Inside the function __parse__, a ```quote``` division is first located by the following query, which will return the whole HTML block under the element ```div``` with class ```quote```

```python
response.css('div.quote')
```

The quote text is then queried by the rule

```python
quote.css('span.text::text')
```

which first asks for the ```span``` element with class of ```text``` and then the ```text``` property of the that element.

The same goes for the field ```author``` and ```tags``` information.

Now you get a general idea about how data is downloaded and how information is extracted based on CSS selector. In addition to CSS selector, there is another query mechanism called XPath, both of which are supported by Scrapy to locate almost any HTML elements on a web page.


# XPath query
XPath is based on path expressions to navigate through elements and attributes in an XML file. For example, the XPath query to locate the last book element of the below HTML block is shown in below.

```html
#xpath query for the last book element
/bookstore/book[last()]
```

```html
<bookstore>

<book category="cooking">
  <title lang="en">Everyday Italian</title>
  <author>Giada De Laurentiis</author>
  <year>2005</year>
  <price>40.00</price>
</book>

<book category="children">
  <title lang="en">Harry Potter</title>
  <author>J K. Rowling</author>
  <year>2005</year>
  <price>29.99</price>
</book>

</bookstore>
```

XPath is also very flexible in selecting children element. For example, the below query searches for  all book elements whose ```price``` child is greater than 35.0

```html
/bookstore/book[price>35.0]
```

The below example shows how to load a html file and then apply an XPath query using Scrapy.

```python
path = './data/bookstore.html'
with open(path, 'r') as file:
    body = file.read()
    book = Selector(text=body).xpath('//bookstore/book[price>35]').get()
    print(book)
```
The query result is pasted in below.

```html
<book category="cooking">
  <title lang="en">Everyday Italian</title>
  <author>Giada De Laurentiis</author>
  <year>2005</year>
  <price>40.00</price>
</book>
```

# Combine CSS selector and XPath

CSS selector, in contrast use patterns to search for styled elements. While XPath query requires a path to the element we want to reach, which could become cumbersome in some cases, CSS is more flexible in a way that elements can be defined in terms of their class or attributes. Therefore, complex query could be more simplified by combining CSS selectors and XPath.

```html
<html>
 <head>
  <base href='http://example.com/' />
  <title>Example website</title>
 </head>
 <body>
  <div id='images'>
   <a href='image1.html'>Name: My image 1 <br /><img src='image1_thumb.jpg' /></a>
   <a href='image2.html'>Name: My image 2 <br /><img src='image2_thumb.jpg' /></a>
   <a href='image3.html'>Name: My image 3 <br /><img src='image3_thumb.jpg' /></a>
   <a href='image4.html'>Name: My image 4 <br /><img src='image4_thumb.jpg' /></a>
   <a href='image5.html'>Name: My image 5 <br /><img src='image5_thumb.jpg' /></a>
  </div>
 </body>
</html>
```
For example, a search for all ```src``` elements in the above HTML file could be carried out by the following query.

```python
response.css('img').xpath('@src').get()
```

The result will be a list of image relative links as follows.

```html
['image1_thumb.jpg',
 'image2_thumb.jpg',
 'image3_thumb.jpg',
 'image4_thumb.jpg',
 'image5_thumb.jpg']
```

# Conclusion
In this note, I explained the general flow of a simple Scrapy Spider and two mechanisms for extracting information from a HTML file. The XPath approach structure a query as a path from the parent nodes to children nodes; whereas CSS selector locates elemenets based on their class or attributes. Both mechanisms could be combined together to form complex queries.

# Reference
[1] https://docs.scrapy.org
