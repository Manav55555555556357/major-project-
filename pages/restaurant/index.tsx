// pages/restaurant-dashboard.tsx
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import styles from './index.module.css';
import { createDonation, auth, getRequestedDonations, acceptDonationRequest, declineDonationRequest, getUserDetails } from '../../utils/firebase';
import { getFoodFreshness } from '../../utils/backend-service';
import Loader from '../../components/Loader';

// Modify the type definition of DonationRequest
type DonationRequest = {
    requestId: string;
    donationId: string;
    ngoId: string;
    ngoName: string;
    foodName: string;
    quantity: number;
};

const RestaurantDashboard: React.FC = () => {
    const [donationRequests, setDonationRequests] = useState<DonationRequest[]>([]);
    const [showModal, setShowModal] = useState(false);
    const [foodName, setFoodName] = useState('');
    const [quantity, setQuantity] = useState(0);
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(true);
    const [updating, setUpdating] = useState(false);

    useEffect(() => {
        fetchDonationRequests();
    }, []);

    const fetchDonationRequests = async () => {
        setLoading(true);
        const restaurantId = auth.currentUser.uid;
        const requestedDonations = await getRequestedDonations(restaurantId);
        console.log(requestedDonations)
        setDonationRequests(requestedDonations);
        setLoading(false);
    };

    const handleCreateDonation = async () => {
        if (!imageFile) {
            alert('Please select an image file');
            return;
        }
        // Replace the restaurantId with the actual restaurant ID
        setLoading(true)
        const restaurantId = auth.currentUser.uid;
        const userDetails = await getUserDetails();
        const pincode = userDetails.data().userData.pinCode;
        const { imageUrl } = await createDonation(restaurantId, foodName, quantity, imageFile);
        const foodFreshness = await getFoodFreshness(new Date(), imageUrl, pincode);
        console.log("Food Freshness: ", foodFreshness)
        setShowModal(false);
        setFoodName('');
        setQuantity(0);
        setImageFile(null);
        alert('Donation created!');
        setLoading(false);
    };

    const handleAcceptRequest = async (requestId: string, donationId: string) => {
        setUpdating(true);
        await acceptDonationRequest(requestId, donationId);
        await fetchDonationRequests();
        alert('Donation request accepted.');
        setUpdating(false);
    };

    const handleDeclineRequest = async (requestId: string, donationId: string) => {
        setUpdating(true);
        await declineDonationRequest(requestId, donationId);
        await fetchDonationRequests();
        alert('Donation request declined.');
        setUpdating(false);
    };

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setImageFile(e.target.files[0]);
        }
    };


    return (
        <div className={styles.container}>
            <h1 className={styles.title}>Restaurant Dashboard</h1>
            <Link href="/login">
                <button className={styles.logoutButton}>Logout</button>
            </Link>
            <div className={styles.requestsContainer}>
                {donationRequests.map((request) => (
                    <div key={request.requestId} className={styles.requestCard}>
                        <h2 className={styles.ngoName}>{request.ngoName}</h2>
                        <ul className={styles.donationList}>
                            <li>
                                {request.foodName} ({request.quantity})
                            </li>
                        </ul>
                        <button
                            className={styles.acceptButton}
                            onClick={() => handleAcceptRequest(request.requestId, request.donationId)}
                        >
                            Accept
                        </button>
                        <button
                            className={styles.declineButton}
                            onClick={() => handleDeclineRequest(request.requestId, request.donationId)}
                        >
                            Decline
                        </button>
                    </div>
                ))}
            </div>
            <button className={styles.createDonationButton} onClick={() => setShowModal(true)}>Create Donation</button>

            {showModal && (
                <div className={styles.modal}>
                    <div className={styles.modalContent}>
                        <h2>Create Donation</h2>
                        <input
                            type="text"
                            placeholder="Food Name"
                            value={foodName}
                            onChange={(e) => setFoodName(e.target.value)}
                        />
                        <input
                            type="number"
                            placeholder="Quantity in kg"
                            value={quantity}
                            onChange={(e) => setQuantity(Number(e.target.value))}
                        />
                        <input
                            type="file"
                            accept="image/*"
                            onChange={handleImageChange}
                        />
                        <button onClick={handleCreateDonation}>Submit</button>
                        <button onClick={() => setShowModal(false)}>Cancel</button>
                    </div>
                </div>
            )}
            {(updating || loading) &&
                <div className={styles.overlay}>
                    <Loader />
                </div>
            }
            <style jsx global>
                {`
                    :root {
                        /* Color palette */
                        --primary-color: #61876E;
                        --secondary-color: #A6BB8D;
                        --tertiary-color: #EAE7B1;
                        --background-color: #3C6255;
                        --text-color: #FFFFFF;
                        --input-background-color: #A6BB8D;
                        --border-color: #EAE7B1;
                        --error-color: #E74C3C;
                    }
                    body {
                        background-color: #3C6255;
                        font-family: sans-serif;
                        margin: 0;
                        color: #EAE7B1;
                    }
                `}
            </style>

        </div>
    );
};

export default RestaurantDashboard;
