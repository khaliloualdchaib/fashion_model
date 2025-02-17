Here's a structured assignment text for your computer vision project:  

---

# **Contour Lab AI Assignment**  
### **Fashion Metadata Inference & Personalized Clothing Recommendation API**  

## **Objective**  
In this assignment, you will develop a machine learning model capable of inferring detailed metadata for clothing items based on an input image, brand name, and barcode. The extracted metadata will then be used to build a recommendation system that suggests the top 10 most suitable clothing items for different body types.  

## **Dataset**  
You are provided with a CSV file containing a dataset of fashion items. Each row represents a clothing item with the following attributes:  

- **image_url_1**: A link to the image of the clothing item  
- **category**: Type of clothing (e.g., shirt, dress, pants)  
- **style**: Style of the clothing (e.g., casual, formal, streetwear)  
- **colors**: Colors of the item (e.g., magenta, white)  
- **gender**: Intended gender for the clothing (e.g., male, female, unisex)  
- **pattern**: Design pattern (e.g., solid, floral, striped)  
- **occasion**: Suitable occasions for wearing the item (e.g., semi-formal, summer)  
- **material, fit, sleeve type, length, neckline type, and more** (full list provided in the dataset)  
- **barcode**: A unique identifier for each clothing item  
- **brand**: The brand of the clothing item  

## **Part 1: Metadata Inference Model**  
1. **Train a model** that, given an input **image, brand, and barcode**, can accurately infer the clothing item's metadata, including style, fit, sleeve type, pattern, and other attributes.  
2. The model should handle **multi-label classification**, as an item may belong to multiple categories (e.g., "casual" and "semi-formal").  
3. Store the inferred metadata for each item in a structured format (e.g., a database).  

## **Part 2: Clothing Recommendation System**  
1. Develop a recommendation algorithm that ranks clothing items based on their suitability for different body types.  
2. The system should support recommendations for the following three body types:  
   - **Hourglass**  
   - **Rectangle**
   - **Apple**  
3. Define a logic for how different attributes (e.g., fit, neckline type, length) influence the suitability of clothing for each body type.  

## **Part 3: API Development**  
1. Build an API endpoint that allows users to **retrieve clothing recommendations** based on body type.  
2. The API should accept **one of the three body types as input** and return the **barcodes of the top 10 most recommended clothing items**.  
3. Example API request:  

   ```http
   GET /recommend?bodytype=hourglass
   ```

   Example API response:  

   ```json
   {
       "recommended_items": [
           "48920502040101",
           "58230601340212",
           "39120409780133",
           ...
       ]
   }
   ```  

## **Evaluation Criteria**  
- **Methodology** of metadata inference  
- **Effectiveness** of the recommendation system
- **Aptitude** to translate fashion rules into code
- **Performance** and **scalability** of the API  
- **Code structure** and **readability**
- **System Level Thinking**

## **Deliverables**  
1. **Trained model** for metadata inference  
2. **Recommendation algorithm implementation**  
3. **API implementation** with a test script or OpenAPI like UI
4. **Short report** explaining the approach, challenges, and improvements

## **Final Notes**
- The recommender will only be used for female shoppers.
- No accessoiries should be recommended.
- Don't spend the majority of your time training the perfect model, methodology is more important.
Good luck! 🚀