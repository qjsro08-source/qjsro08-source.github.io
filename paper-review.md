---
layout: archive
title: "Paper Review"
permalink: /paper-review/
author_profile: true
toc: true
---

{% assign entries_layout = page.entries_layout | default: 'list' %}
{% assign posts = site.categories['Paper Review'] %}

{% for post in posts %}
  {% include archive-single.html type=entries_layout %}
{% endfor %}