CREATE USER john IDENTIFIED BY '1234';

SELECT *
FROM mysql.user;

CREATE USER bob@aman.com IDENTIFIED BY '12';

DROP USER bob@aman.com;

SET PASSWORD FOR john@localhost = '1234';

GRANT SELECT,UPDATE, DELETE,EXECUTE,INSERT
ON sql_store.*
TO john;

REVOKE UPDATE, DELETE,EXECUTE,INSERT
ON sql_store.*
FROM john;

SELECT *
FROM mysql.user;

USE sql_store;

SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

START TRANSACTION;
SELECT points FROm customers WHERE customer_id = 1;
SELECT points FROm customers WHERE customer_id = 1;
COMMIT;

USE Sql_store;

EXPLAIN
SELECT customer_id
FROM customers
WHERE state <> 'CA';

CREATE INDEX idx_state ON customers(state);

EXPLAIN  
SELECT customer_id
FROM customers
WHERE points > 1000;


CREATE INDEX idx_points ON customers(points);
show INDEXES in customers;

CREATE INDEX idx_lastname ON customers (last_name(20));

SHOW INDEXES IN customers;

DROP INDEX idx_points ON customers;

use sql_invoicing;

DROP procedure IF EXISTS make_payments;

DELIMITER $$
CREATE procedure make_payments
(
	invoice_id INT,
    payment_amount DECIMAL(9,2),
    payment_date DATE
)
BEGIN
	IF payment_amount <=0 THEN
		SIGNAL sqlstate '22003'
        SET message_text = 'Invalid Payment amount';
	END IF;
	
    UPDATE invoices i
    SET i.payment_total = payment_amount,
		i.payment_date = payment_date
	WHERE i.invoice_id = invoice_id;
END$$
DELIMITER ;

CALL make_payments(2,-100,'2019-01-01');

DROP procedure IF EXISTS get_unpaid_invoices_for_clients;

DELIMITER $$
CREATE procedure get_unpaid_invoices_for_clients
(
	client_id INT,
    OUT invoices_count INT,
    OUT invoices_total DECIMAL(9,2)
)
BEGIN
	SELECT COUNT(*), SUM(invoices_total)
    INTO invoices_count, invoices_total
    FROM invoices i
    WHERE i.client_id = client_id 
		AND payment_total = 0;
END$$
DELIMITER ;

CALL get_unpaid_invoices_for_clients(2);

DROP procedure IF EXISTS get_risk_factor;
DELIMITER $$
CREATE procedure get_risk_factor()
BEGIN
	DECLARE risk_factor DECIMAL(9,2) DEFAULT 0;
    DECLARE invoices_total DECIMAL(9,2);
    DECLARE invoices_count INT;
    
    SELECT COUNT(*), SUM(invoice_total)
    INTO invoices_count,invoices_total
    FROM invoices;
    
    SET risk_factor = 100.3;
    
    SELECT risk_factor;
    
END$$
DELIMITER ;

CALL get_risk_factor();

DROP FUNCTION IF EXISTS get_risk_factor_for_client;

DELIMITER $$
CREATE FUNCTION get_risk_factor_for_client
(
	client_id INT
)
RETURNS INTEGER
READS SQL DATA
BEGIN
	DECLARE risk_factor DECIMAL(9,2) DEFAULT 0;
    DECLARE invoices_total DECIMAL(9,2);
    DECLARE invoices_count INT;
    
	SELECT COUNT(*), SUM(invoice_total)
    INTO invoices_count,invoices_total
    FROM invoices i
    WHERE i.client_id = client_id;
    
    SET risk_factor = IFNULL(invoices_total/invoices_count * 5,0);
    
RETURN risk_factor;
END$$
DELIMITER ;

SELECT
	client_id,
    name,
    get_risk_factor_for_client(client_id)
FROM clients;

USE sql_inventory;

SELECT *
FROM products
WHERE unit_price > (SELECT unit_price
					FROM products
					WHERE product_id = 3);
                    
USE sql_hr;

SELECT *
FROM employees
WHERE salary > (
				SELECT AVG(salary)
				FROM employees);
                
USE sql_store;

SELECT *
FROM products
WHERE product_id NOT IN (
						SELECT DISTINCT product_id
						FROM order_items);
                        
USE sql_store;

SELECT o.customer_id,
	c.first_name,
    c.last_name
FROM orders o
JOIN customers c
	USING (customer_id)
JOIN  order_items oi
	USING (order_id)
WHERE oi.product_id = 3;

USE sql_invoicing;

SELECT *
FROM invoices
WHERE invoice_total > ALL (
						SELECT invoice_total
						FROM invoices
						WHERE client_id = 3);
                        
USE sql_invoicing;

SELECT * 
FROM invoices i
WHERE invoice_total > (
						SELECT AVG(invoice_total)
						FROM invoices
						WHERE client_id = i.client_id);
                        
                        
SELECT *
FROM clients c
WHERE EXISTS (
				SELECT client_id
				FROM invoices
				WHERE client_id = c.client_id);
                
SELECT *
FROM products p
WHERE NOT EXISTS (
				SELECT DISTINCT product_id
				FROM order_items
                WHERE product_id = p.product_id);
                
SELECT *
FROM products p
WHERE product_id NOT IN (
							SELECT DISTINCT product_id
							FROM order_items);
                            

USE sql_invoicing;

SELECT *
FROM (
		SELECT  client_id,
				name,
				(SELECT SUM(invoice_total)
					FROM invoices
					WHERE client_id = c.client_id ) AS 'SALES_PER_CLIENT',
				(SELECT AVG(invoice_total)
					FROM invoices) AS 'AVG_CLIENT_SALES',
				(SELECT AVG_CLIENT_SALES) - (SELECT SALES_PER_CLIENT) AS 'DEVIATION' 
		FROM clients c
) AS SALES_SUMMARY
WHERE SALES_PER_CLIENT IS NOT NULL; 

SELECT TRIM('   PISS ON ME ');

SELECT SUBSTRING('PISS_ON_ME',3);

SELECT LENGTH('PISS_ON_ME');

SELECT (SELECT LOWER('PISS_ON_ME')) AS 'piss',
		(SELECT UPPER(piss)) AS 'upper';

SELECT LOCATE('_','PISS_ON_ME');

SELECT NOW();

SELECT LOCATE(2,YEAR(NOW()));

SELECT EXTRACT(MONTH FROM NOW());

SELECT 	UPPER(MONTHNAME(NOW())) AS 'a',
		LENGTH(MONTHNAME(NOW())) AS 'b',
        CONCAT((SELECT a),' ',(SELECT b));

SELECT curdate();

SELECT curtime();
        
SELECT date_format(NOW(), '%Y-%m-%d-%H-%i-%S');

SELECT DATE_ADD(NOW(), INTERVAL 3 HOUR);
SELECT DATE_SUB(NOW(), INTERVAL 1 DAY);

SELECT DATEDIFF('2019-03-09','2019-03-05');

SELECT CEILING((TIME_TO_SEC('11:00') - TIME_TO_SEC('10:00'))/60);

USE Sql_store;

SELECT order_id,
		coalesce(shipper_id,comments,'Piss on me')
FROM orders;

SELECT order_id,
		order_date,
        IF(
			YEAR(order_date) = YEAR(DATE_SUB(NOW(),INTERVAL 4 YEAR)),
            'Active',
            'Archived')
FROM orders;

SELECT order_id,
		order_date,
        CASE 
			WHEN YEAR(order_date) = YEAR(DATE_SUB(NOW(),INTERVAL 4 YEAR)) THEN 'Active'
            WHEN YEAR(order_date) = YEAR(DATE_SUB(NOW(),INTERVAL 5 YEAR)) THEN 'Overdue'
            ELSE 'Pee on them .. woof woof'
		END AS 'Test'
FROM orders;

USE sql_invoicing;

CREATE OR REPLACE VIEW sales_by_client AS
SELECT 
	c.client_id,
    c.name,
    SUM(invoice_total) AS total_sales
FROM clients c
JOIN invoices i USING(client_id)
GROUP BY c.client_id;

CREATE OR REPLACE VIEW client_balance AS
SELECT 
	c.client_id,
    c.name,
    SUM(invoice_total - payment_total) AS total_sales
FROM clients c
JOIN invoices i USING(client_id)
GROUP BY c.client_id;

SELECT *
FROM sales_by_client
JOIN clients USING(client_id)
ORDER BY total_sales DESC;

SELECT *
FROM client_balance;

CREATE OR REPLACE VIEW invoices_with_balance AS
SELECT invoice_id,
	number,
    client_id,
    invoice_total,
    payment_total,
    invoice_total - payment_total AS 'balance',
    invoice_date,
    due_date,
    payment_date
FROM invoices
WHERE (invoice_total - payment_total) > 0
WITH CHECK OPTION;


SELECT count(invoice_id)
FROM invoices
GROUP BY client_id
HAVING count(invoice_id) > 1
ORDER BY count(invoice_id) DESC;

DELIMITER $$
CREATE procedure get_clients()
BEGIN
	SELECT * FROM clients;
END$$
DELIMITER ;

CALL get_clients();

DELIMITER $$
CREATE procedure get_invoices_with_balance()
begin
	SELECT *
    FROM invoices
    WHERE payment_total > 0;
END$$
DELIMITER ;

CALL get_invoices_with_balance();

DROP procedure IF EXISTS get_clients_by_state;

DELIMITER $$
CREATE procedure get_clients_by_state
(
	state CHAR(2)
)
BEGIN
IF state IS NULL THEN
	SET state = 'CA';
END IF;
	SELECT *
    FROM clients c
    WHERE c.state = state;
END$$
DELIMITER ;

CALL get_clients_by_state(NULL);

DROP procedure IF EXISTS get_payments;

DELIMITER $$
CREATE procedure get_payments
(
	client_id INT,
    payment_method TINYINT
)
BEGIN
	SELECT *
    FROM  payments p
    WHERE p.client_id = IFNULL(client_id,p.client_id)
		AND p.payment_method = IFNULL(payment_method,p.payment_method);
END$$
DELIMITER ;

call get_payments(5,1)

USE sql_store;

UPDATE customers
SET points = points + 50
WHERE birth_date < '1990-01-01';

UPDATE orders
SET comments = 'GOLD!'
WHERE customer_id IN (
					SELECT customer_id
					FROM customers
					WHERE points > 3000);

USE sql_invoicing;                    

DELETE FROM invoices
WHERE client_id = (
					SELECT client_id
					FROM clients
					WHERE name = 'Myworks'
					);

SELECT client_id
FROM clients
WHERE name = 'Myworks';

SELECT  MAX(invoice_total) AS 'highest',
		MIN(invoice_total) AS 'lowest',
        AVG(invoice_total) AS 'average',
        SUM(invoice_total) AS 'sum',
        count(invoice_total) AS 'count'
FROM invoices;

USE sql_invoicing;                    

SELECT 'First half of 2019' AS 'date_range',
		SUM(invoice_total) AS 'Total_Invoice',
		SUM(payment_total) AS 'Total Payments'
FROM invoices
WHERE invoice_date BETWEEN '2019-01-01' AND '2019-06-30'
UNION
SELECT 'Second half of 2019' AS 'date_range',
		SUM(invoice_total) AS 'Total_Invoice',
		SUM(payment_total) AS 'Total Payments'
FROM invoices
WHERE invoice_date BETWEEN '2019-07-01' AND '2019-12-31'
UNION
SELECT 'Total' AS 'date_range',
		SUM(DISTINCT invoice_total) AS 'Total_Invoice',
		SUM(payment_total) AS 'Total Payments'
FROM invoices
WHERE invoice_date BETWEEN '2019-01-01' AND '2019-12-31' ;

SELECT client_id,
		SUM(invoice_total) as 'invoices'
FROM invoices
WHERE invoice_date >='2019-07-01'
GROUP BY client_id WITH ROLLUP
ORDER BY invoices DESC;

SELECT p.date,
		pm.name,
        sum(p.amount) AS 'total_payments'
FROM payments p
JOIN payment_methods pm
ON p.payment_method = pm.payment_method_id
GROUP BY p.date, pm.name
HAVING total_payments > 5
ORDER BY total_payments DESC;

USE sql_store;

START TRANSACTION;

INSERT INTO orders (customer_id, order_date, status)
VALUES (1,'2019-01-01',1);

INSERT INTO order_items
VALUES (LAST_INSERT_ID(),1,1,1);

COMMIT;

