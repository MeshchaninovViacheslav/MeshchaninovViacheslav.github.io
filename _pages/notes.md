---
layout: page
title: notes
permalink: /notes/
description: A growing collection of your cool notes.
nav: true
nav_order: 3
horizontal: false
---

<!-- pages/notes.md -->
<div class="notes">
{% assign notes_by_category = site.notes | group_by_exp: "item", "item.path | split: '/' | slice: 1, 1 | join: ''" %}
{% for category_group in notes_by_category %}
  {% if category_group.name == "" or category_group.name == nil %}{% continue %}{% endif %}
  <hr class="my-5">
  <a id="{{ category_group.name | slugify }}" href=".#{{ category_group.name | slugify }}">
    <h2 class="category">{{ category_group.name }}</h2>
  </a>

  {% assign notes_with_topics = "" | split: "" %}
  {% for note in category_group.items %}
    {% if note.path | split: '/' | size > 3 %}
      {% assign notes_with_topics = notes_with_topics | push: note %}
    {% endif %}
  {% endfor %}

  {% if notes_with_topics.size > 0 %}
    {% assign topics = notes_with_topics | group_by_exp: "item", "item.path | split: '/' | slice: 2, 1 | join: ''" %}
    <div class="row row-cols-1 row-cols-md-3">
      {% for topic in topics %}
        {% include topic_card.liquid topic=topic %}
      {% endfor %}
    </div>
  {% endif %}
{% endfor %}
</div>