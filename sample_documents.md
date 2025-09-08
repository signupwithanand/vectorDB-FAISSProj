# Sample Documents for FAISS Demo

Here are some sample documents you can copy and paste into the app to quickly see how FAISS works:

## Technology Documents
```
Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.

Deep learning uses neural networks with multiple layers to process complex patterns in data, making it powerful for image recognition and natural language processing.

Python is a versatile programming language widely used in data science, web development, and artificial intelligence applications.

FAISS is a library for efficient similarity search and clustering of dense vectors, developed by Facebook AI Research.

Natural language processing enables computers to understand, interpret, and generate human language in a meaningful way.
```

## Science Documents
```
The theory of relativity revolutionized our understanding of space, time, and gravity in the early 20th century.

Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.

DNA contains the genetic instructions for the development and function of all living organisms on Earth.

Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities.

The periodic table organizes chemical elements based on their atomic number and recurring chemical properties.
```

## Cooking Documents
```
Italian pasta should be cooked al dente, meaning it's firm to the bite and not overcooked or mushy.

Baking requires precise measurements and temperatures to ensure proper chemical reactions between ingredients.

Fresh herbs like basil, oregano, and thyme add incredible flavor to Mediterranean dishes.

Slow cooking methods like braising help break down tough meat fibers, resulting in tender and flavorful dishes.

Proper knife skills are essential for safe and efficient food preparation in any kitchen.
```

## Sample Search Queries to Try

After adding the documents above, try these search queries to see how FAISS finds similar content:

- "artificial intelligence and computers" (should match ML/AI docs)
- "cooking techniques and food" (should match cooking docs)
- "scientific discoveries and research" (should match science docs)
- "programming languages for data" (should match Python/tech docs)
- "plants and biological processes" (should match photosynthesis/DNA docs)

## What to Observe

1. **Embedding Similarity**: Notice how documents with similar topics cluster together in the 2D visualization
2. **Search Relevance**: See how queries match documents based on semantic meaning, not just keyword matching
3. **Similarity Scores**: Higher scores (closer to 1.0) indicate more similar content
4. **Visual Clustering**: In the scatter plot, similar documents appear closer together
5. **Query Positioning**: Your search query (green star) appears near relevant documents

## Learning Exercises

1. **Add Contradictory Documents**: Add documents on the same topic but with different viewpoints
2. **Test Edge Cases**: Try very short vs. very long documents
3. **Experiment with Queries**: Use synonyms, technical terms, or casual language
4. **Compare Reduction Methods**: Switch between PCA and t-SNE to see different visualizations
5. **Build Topic Collections**: Add 5-10 documents on the same specific topic and see how they cluster