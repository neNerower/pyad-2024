def personal_recs(user_id, ratingsDf, booksDf, svd, reg):
    unrated_book_ids = ratingsDf[(ratingsDf['User-ID'] == user_id) & (ratingsDf['Book-Rating'] == 0)]["ISBN"].unique()

    # SVD personal recommendations
    recommended_book_ids = [predict.iid for isbn in unrated_book_ids if (predict := svd.predict(user_id, isbn)).est >= 8]
    recommended_books = booksDf[booksDf['ISBN'].isin(recommended_book_ids)]
    recommended_books = recommended_books[["ISBN", "Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"]]

    # Get most interesting by regression
    reg_rating_predict = reg.predict(recommended_books)
    recommended_books['Predicted-Raiting'] = reg_rating_predict

    return recommended_books.sort_values('Predicted-Raiting', ascending=False).head(10)


#############################################################################################################################################################
# Топ 10 рекомендованных книг для пользователя (198711)
#############################################################################################################################################################
# ISBN 	        Book-Title 	                            Book-Author 	    Publisher 	                            Year-Of-Publication 	Predicted-Raiting
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# 0439064872    harry potter chamber secrets book 	    J. K. Rowling 	    Scholastic 	                            2000.0 	                9.111272
# 0064409422 	lion witch wardrobe collector edition 	C. S. Lewis 	    HarperTrophy 	                        2000.0 	                8.928080
# 0836218353 	yukon ho 	                            Bill Watterson 	    Andrews McMeel Publishing 	            1989.0 	                8.724192
# 0394900014 	cat hat read beginner books 	        Seuss 	            Random House Books for Young Readers 	1957.0 	                8.611320
# 0345339681 	hobbit enchanting prelude lord rings 	J.R.R. TOLKIEN 	    Del Rey 	                            1986.0 	                8.418014
# 0064405052 	magician nephew narnia 	                C. S. Lewis 	    HarperTrophy 	                        1994.0 	                8.413391
# 0064400557 	charlotte web trophy newbery 	        E. B. White 	    HarperTrophy 	                        1974.0 	                8.341128
# 0684801221 	old man sea 	                        Ernest Hemingway 	Scribner 	                            1995.0 	                8.340857
# 0064471101 	magician nephew rack narnia 	        C. S. Lewis 	    HarperCollins 	                        2002.0 	                8.261162
# 0440498058 	wrinkle time 	                        MADELEINE L'ENGLE 	Yearling 	                            1998.0 	                8.244678	                        
