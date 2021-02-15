
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

class Solution:
    def __init__(self) -> None:      
        file = 'data/chipotle.tsv'
        self.chipo =pd.read_csv(file, sep = '\t')

    def top_x(self, count) -> None:          
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:      
        return self.chipo.order_id.count()
    
    def info(self) -> None:        
        print(self.chipo.info())        
        pass
    
    def num_column(self) -> int:       
        return len(self.chipo.columns)
    
    def print_columns(self) -> None:              
        print(self.chipo.columns)
        pass
    
    def most_ordered_item(self):               
        grouped_Item = self.chipo.groupby("item_name")["quantity"].sum().sort_values().tail(1).to_dict()
        item_name = list(grouped_Item.keys())[0]        
        quantity = self.chipo.groupby("choice_description")["quantity"].sum().sort_values().max()                
        order_id = self.chipo.groupby("item_name")["order_id"].sum()[item_name]                                       
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:             
        return self.chipo["quantity"].sum()
   
    def total_sales(self) -> float:           
        total_sales = (self.chipo["item_price"].apply(lambda x: x.strip('$')).astype(float) 
                      * self.chipo["quantity"]).sum()                       
        return total_sales             
   
    def num_orders(self) -> int:        
        return self.chipo["order_id"].nunique()
    
    def average_sales_amount_per_order(self) -> float:                
        average_sales_amount_per_order = (self.chipo["item_price"].apply(lambda x: x.strip('$')).astype(float) * self.chipo["quantity"]).sum()                                  
        average_sales_amount_per_order /= self.chipo["order_id"].nunique()
        average_sales_amount_per_order = round(average_sales_amount_per_order, 2)        
        return average_sales_amount_per_order

    def num_different_items_sold(self) -> int:            
        return self.chipo["item_name"].nunique()
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)                
        df = pd.DataFrame.from_dict(letter_counter, orient='index').reset_index()
        df.columns = ["item_name", "quantity"]     
        df.rename(columns={"index" : "item_name", 0 : "quantity"})
        df = df.sort_values("quantity", ascending=False).head(5)        
        fig = plt.figure()
        axes = fig.add_axes([0,0,1,1])        
        axes.bar(df["item_name"], df["quantity"])
        plt.title("Most popular items")
        plt.xlabel("Items")
        plt.ylabel("Number of Orders")
        plt.show(block = True)      
        pass
        
    def scatter_plot_num_items_per_order_price(self) -> None:                
        self.chipo["item_price"] = self.chipo["item_price"].apply(lambda x: x.strip('$')).astype(float) * self.chipo["quantity"]                                       
        xData = self.chipo.groupby("order_id")["item_price"].sum()
        yData = self.chipo.groupby("order_id")["quantity"].sum()                
        plt.scatter(x = xData, y = yData, s = 50, c = "blue")
        plt.title("Numer of items per order price")
        plt.xlabel("Order Price")
        plt.ylabel("Num Items")
        plt.show()
        pass 
        

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926	
    assert quantity == 159
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
